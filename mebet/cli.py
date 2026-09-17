"""Command line interface.

Everything the UI can do is available here, which keeps the application
scriptable and makes the data layer testable without a browser.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys

from .backtest import BacktestConfig, BacktestRunner
from .config import get_settings
from .db.session import init_db, session_scope
from .engine.service import AnalysisRequest, AnalysisService
from .logging_setup import setup_logging
from .pipeline import AcquisitionPipeline, season_label, seasons_back


def cmd_init(args) -> int:
    init_db()
    print(f"database ready at {get_settings().resolved_database_url()}")
    return 0


def cmd_sources(args) -> int:
    with session_scope() as s:
        for row in AnalysisService(s).source_status():
            mark = "OK  " if row["available"] else "DOWN"
            print(f"[{mark}] {row['key']:24} {row['detail']}")
    return 0


def cmd_ingest(args) -> int:
    init_db()
    current = season_label(dt.date.today())
    seasons = seasons_back(current, args.seasons)
    with session_scope() as s:
        pipeline = AcquisitionPipeline(s)
        for competition in args.competition:
            print(f"\n=== {competition} ===")
            report = pipeline.load_history(competition, seasons)
            pipeline.load_fixtures(competition, report)
            if args.players:
                pipeline.load_player_stats(competition, seasons[-2:], report)
            pipeline.load_availability(competition, report)
            for outcome in report.outcomes:
                status = "ok  " if outcome.ok else "FAIL"
                detail = (f"{outcome.records} records, {outcome.written} new, "
                          f"{outcome.updated} updated") if outcome.ok else outcome.error[:90]
                print(f"  [{status}] {outcome.source_key:22} {outcome.capability:26} {detail}")
            if report.conflicts:
                print(f"  conflicts: {len(report.conflicts)}")
    return 0


def cmd_analyze(args) -> int:
    init_db()
    kickoff = None
    date = None
    if args.kickoff:
        kickoff = dt.datetime.fromisoformat(args.kickoff)
    if args.date:
        date = dt.date.fromisoformat(args.date)
    with session_scope() as s:
        service = AnalysisService(s)
        response = service.analyze(AnalysisRequest(
            competition_key=args.competition, home_team=args.home, away_team=args.away,
            kickoff=kickoff, kickoff_date=date, refresh=not args.no_refresh,
            history_seasons=args.seasons, include_players=not args.no_players,
        ))
        if args.json:
            print(json.dumps(response.as_dict(), indent=2, default=str))
            return 0 if response.ok else 1
        _print_prediction(response)
        return 0 if response.ok else 1


def _print_prediction(response) -> None:
    if not response.ok:
        print(f"could not analyse: {response.message}")
        return
    p = response.prediction
    width = 68
    print("=" * width)
    print(f"{p.home_team}  vs  {p.away_team}")
    print(f"{p.competition} | {p.kickoff or '(kickoff time unknown)'}")
    print(f"generated {p.generated_at:%Y-%m-%d %H:%M} UTC | version {p.version}")
    print("=" * width)
    if p.outcome:
        print("\nMATCH RESULT")
        for key, label in (("home", p.home_team), ("draw", "Draw"), ("away", p.away_team)):
            bar = "#" * int(round(p.outcome[key] * 40))
            print(f"  {label[:22]:24} {p.outcome[key]:6.1%}  {bar}")
        print(f"\n  Most likely score: {p.most_likely_score}")
    for warning in p.warnings:
        print(f"\n  ! {warning}")

    markets = p.targets_by_market()
    def show(market, title):
        rows = markets.get(market)
        if not rows:
            return
        print(f"\n{title}")
        for r in rows:
            if not r["sufficient_data"]:
                print(f"  {r['selection']:14} unavailable - {r['note']}")
                continue
            value = (f"{r['probability']:.1%}" if r["probability"] is not None
                     else str(r["expected_value"]))
            name = r["subject_name"] or r["selection"]
            print(f"  {name[:26]:28} {value}")

    show("expected_goals", "GOALS")
    show("over_under_2.5", "OVER/UNDER 2.5 GOALS")
    show("btts", "BOTH TEAMS TO SCORE")
    show("expected_corners", "CORNERS")
    show("expected_cards", "CARDS")

    if p.quality:
        print("\nDATA QUALITY")
        for line in p.quality.summary_lines():
            print(f"  {line}")
        for issue in p.quality.issues:
            if issue.severity in {"warning", "serious"}:
                print(f"  ! [{issue.severity}] {issue.message}")

    if p.factors:
        print("\nWHY")
        for factor in p.factors[:8]:
            print(f"  - {factor.statement}")
    print()


def cmd_backtest(args) -> int:
    init_db()
    with session_scope() as s:
        config = BacktestConfig(
            competition_key=args.competition,
            from_date=dt.date.fromisoformat(args.from_date),
            to_date=dt.date.fromisoformat(args.to_date),
            refit_days=args.refit_days,
            seasons_back=args.seasons_back,
        )
        result = BacktestRunner(s).run(config)
        if args.json:
            print(json.dumps(result.as_dict(), indent=2, default=str))
            return 0
        print(f"\nBacktest: {config.label()}  ({result.n_matches} matches scored)")
        print(f"Baselines: {({k: round(v, 4) for k, v in result.baselines.items()})}")
        print(f"\n{'model':16}{'n':>6}{'accuracy':>10}{'log loss':>11}{'brier':>9}{'calib err':>11}")
        for score in sorted([x for x in result.scores if x.market == "1x2"],
                            key=lambda x: x.metrics.get("log_loss", 9)):
            m = score.metrics
            print(f"{score.model_key:16}{score.n:>6}{m['accuracy']:>10.3f}"
                  f"{m['log_loss']:>11.4f}{m['brier']:>9.4f}{m['calibration_error']:>11.4f}")
        print(f"\nMeasured ensemble weights: {result.ensemble_weights}")
        return 0


def cmd_serve(args) -> int:
    import uvicorn

    settings = get_settings()
    host = args.host or settings.host
    port = args.port or settings.port
    print(f"mebet running at http://{host}:{port}")
    uvicorn.run("mebet.api.app:app", host=host, port=port, reload=args.reload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mebet", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="create the database schema").set_defaults(func=cmd_init)
    sub.add_parser("sources", help="check which data sources are reachable").set_defaults(
        func=cmd_sources
    )

    p = sub.add_parser("ingest", help="download and store historical data")
    p.add_argument("competition", nargs="+", help="competition keys, e.g. ENG.1 ESP.1")
    p.add_argument("--seasons", type=int, default=8)
    p.add_argument("--players", action="store_true", help="also load player statistics")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("analyze", help="analyse one match")
    p.add_argument("--competition", required=True)
    p.add_argument("--home", required=True)
    p.add_argument("--away", required=True)
    p.add_argument("--date", help="YYYY-MM-DD")
    p.add_argument("--kickoff", help="ISO datetime, e.g. 2026-09-20T15:00")
    p.add_argument("--seasons", type=int, default=8)
    p.add_argument("--no-refresh", action="store_true", help="use stored data only")
    p.add_argument("--no-players", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("backtest", help="evaluate models on historical matches")
    p.add_argument("--competition", required=True)
    p.add_argument("--from", dest="from_date", required=True)
    p.add_argument("--to", dest="to_date", required=True)
    p.add_argument("--refit-days", type=int, default=30)
    p.add_argument("--seasons-back", type=int, default=5)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("serve", help="run the web application")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv=None) -> int:
    setup_logging()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

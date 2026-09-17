"""Team identity resolution across sources.

Sources spell the same club differently: "Man United", "Man Utd",
"Manchester United FC". Worse, some pairs share no characters at all -
"Tottenham" and "Spurs" - so string similarity alone would split one club's
history into two, quietly halving every sample size drawn from it.

The resolver therefore works in decreasing order of confidence:

1. exact match on a normalised form;
2. a curated alias map for clubs known to be spelled irreconcilably;
3. aliases learned in earlier runs and persisted to the database;
4. fuzzy similarity, but only above a high threshold *and* with a token in
   common, to avoid merging "Real Madrid" into "Real Sociedad";
5. otherwise a new club is created and the event is logged, because inventing
   a match is worse than admitting a new name was seen.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Iterable, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..logging_setup import get_logger
from ..db.models import Team, TeamAlias

log = get_logger("normalize.teams")

#: Club-name suffixes and prefixes that carry no identifying information.
_NOISE_TOKENS = {
    "fc", "afc", "cf", "sc", "ac", "as", "ss", "ssc", "sv", "vfl", "vfb", "fk",
    "cd", "ud", "rc", "rcd", "club", "calcio", "bk", "if", "spa", "the",
}

#: Punctuation that sits *inside* a word and should vanish rather than split
#: it: "A.C. Milan" must normalise the same way as "AC Milan", and
#: "Nott'm Forest" as "Nottm Forest".
_INTRA_WORD_PUNCT = re.compile(r"[.'\u2019\u02bc`]", re.UNICODE)
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES = re.compile(r"\s+")


def normalise_name(name: str) -> str:
    """Casefold, strip accents and punctuation, drop uninformative tokens."""
    if not name:
        return ""
    text = unicodedata.normalize("NFKD", name)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    text = _INTRA_WORD_PUNCT.sub("", text)
    text = _PUNCT.sub(" ", text)
    tokens = [t for t in _SPACES.sub(" ", text).strip().split(" ") if t]
    kept = [t for t in tokens if t not in _NOISE_TOKENS]
    return " ".join(kept or tokens)


#: Canonical name -> spellings that normalisation cannot reconcile.
#: Only clubs whose variants genuinely differ are listed; everything else is
#: handled by normalisation and needs no entry.
CURATED_ALIASES: dict[str, tuple[str, ...]] = {
    # England
    "Manchester United": ("man united", "man utd", "manchester utd", "man u"),
    "Manchester City": ("man city", "manchester city"),
    "Tottenham Hotspur": ("tottenham", "spurs", "tottenham hotspur"),
    "Wolverhampton Wanderers": ("wolves", "wolverhampton", "wolverhampton wanderers"),
    "Nottingham Forest": ("nott m forest", "nottm forest", "nottingham forest", "nott forest"),
    "Newcastle United": ("newcastle", "newcastle united", "newcastle utd"),
    "Brighton & Hove Albion": ("brighton", "brighton hove albion", "brighton and hove albion"),
    "West Ham United": ("west ham", "west ham united", "west ham utd"),
    "Leeds United": ("leeds", "leeds united"),
    "Leicester City": ("leicester", "leicester city"),
    "Sheffield United": ("sheffield united", "sheffield utd", "sheff united", "sheff utd"),
    "Sheffield Wednesday": ("sheffield weds", "sheffield wednesday", "sheff wed"),
    "West Bromwich Albion": ("west brom", "west bromwich albion", "west bromwich"),
    "Queens Park Rangers": ("qpr", "queens park rangers"),
    "Bournemouth": ("bournemouth", "afc bournemouth"),
    "Stoke City": ("stoke", "stoke city"),
    "Hull City": ("hull", "hull city"),
    "Cardiff City": ("cardiff", "cardiff city"),
    "Swansea City": ("swansea", "swansea city"),
    "Norwich City": ("norwich", "norwich city"),
    "Ipswich Town": ("ipswich", "ipswich town"),
    "Luton Town": ("luton", "luton town"),
    "Birmingham City": ("birmingham", "birmingham city"),
    "Derby County": ("derby", "derby county"),
    "Middlesbrough": ("middlesbrough", "middlesboro"),
    # Spain
    "Atlético Madrid": ("ath madrid", "atletico madrid", "atl madrid", "atletico de madrid"),
    "Athletic Bilbao": ("ath bilbao", "athletic bilbao", "athletic club"),
    "Real Betis": ("betis", "real betis"),
    "Real Sociedad": ("sociedad", "real sociedad"),
    "Celta Vigo": ("celta", "celta vigo"),
    "Real Valladolid": ("valladolid", "real valladolid"),
    "Deportivo Alavés": ("alaves", "deportivo alaves"),
    "Espanyol": ("espanol", "espanyol", "rcd espanyol"),
    "Real Madrid": ("real madrid",),
    "Barcelona": ("barcelona", "fc barcelona"),
    "Rayo Vallecano": ("vallecano", "rayo vallecano"),
    # Italy
    "Internazionale": ("inter", "internazionale", "inter milan"),
    "AC Milan": ("milan", "ac milan"),
    "AS Roma": ("roma", "as roma"),
    "Hellas Verona": ("verona", "hellas verona"),
    "Juventus": ("juventus", "juve"),
    # Germany
    "Bayern Munich": ("bayern munich", "bayern", "fc bayern munchen", "bayern munchen"),
    "Borussia Dortmund": ("dortmund", "borussia dortmund", "bvb"),
    "Borussia Mönchengladbach": ("m gladbach", "mgladbach", "monchengladbach",
                                 "borussia monchengladbach", "gladbach"),
    "Bayer Leverkusen": ("leverkusen", "bayer leverkusen"),
    "Eintracht Frankfurt": ("ein frankfurt", "eintracht frankfurt", "frankfurt"),
    "RB Leipzig": ("rb leipzig", "leipzig"),
    "FC Köln": ("fc koln", "koln", "cologne", "1 fc koln"),
    "Hertha Berlin": ("hertha", "hertha berlin"),
    "Schalke 04": ("schalke", "schalke 04"),
    "Werder Bremen": ("werder bremen", "bremen"),
    "TSG Hoffenheim": ("hoffenheim", "tsg hoffenheim"),
    "VfB Stuttgart": ("stuttgart", "vfb stuttgart"),
    "Mainz 05": ("mainz", "mainz 05"),
    # France
    "Paris Saint-Germain": ("paris sg", "paris saint germain", "psg", "paris"),
    "Olympique Marseille": ("marseille", "olympique marseille", "om"),
    "Olympique Lyonnais": ("lyon", "olympique lyonnais", "ol"),
    "AS Monaco": ("monaco", "as monaco"),
    "Saint-Étienne": ("st etienne", "saint etienne"),
    "Paris FC": ("paris fc",),
}

#: Minimum similarity for a fuzzy match to be accepted.
FUZZY_THRESHOLD = 0.90


class TeamResolver:
    """Maps free-text team names onto stable ``Team`` rows."""

    def __init__(self, session: Session, sport: str = "football") -> None:
        self.session = session
        self.sport = sport
        self._cache: dict[str, int] = {}
        self._curated: dict[str, str] = {}
        for canonical, variants in CURATED_ALIASES.items():
            self._curated[normalise_name(canonical)] = canonical
            for v in variants:
                self._curated[normalise_name(v)] = canonical
        self.unresolved: list[str] = []
        self.fuzzy_matches: list[tuple[str, str, float]] = []

    # -- lookup helpers ---------------------------------------------------
    def _alias_lookup(self, norm: str) -> Optional[int]:
        row = self.session.execute(
            select(TeamAlias).where(TeamAlias.sport == self.sport, TeamAlias.alias_norm == norm)
        ).scalars().first()
        return row.team_id if row else None

    def _existing_teams(self) -> list[Team]:
        return list(
            self.session.execute(select(Team).where(Team.sport == self.sport)).scalars()
        )

    def _fuzzy(self, norm: str) -> Optional[tuple[Team, float]]:
        teams = self._existing_teams()
        if not teams:
            return None
        candidates = {normalise_name(t.canonical_name): t for t in teams}
        # Learned aliases widen the candidate pool.
        for alias in self.session.execute(
            select(TeamAlias).where(TeamAlias.sport == self.sport)
        ).scalars():
            candidates.setdefault(alias.alias_norm, self.session.get(Team, alias.team_id))

        best_name, best_score = None, 0.0
        target_tokens = set(norm.split())
        for cand in candidates:
            score = difflib.SequenceMatcher(None, norm, cand).ratio()
            if score > best_score:
                best_name, best_score = cand, score

        if best_name is None or best_score < FUZZY_THRESHOLD:
            return None
        # Require a shared token so that high character overlap alone
        # ("Real Madrid" / "Real Sociedad") cannot merge two clubs.
        if not (target_tokens & set(best_name.split())):
            return None
        team = candidates[best_name]
        return (team, best_score) if team is not None else None

    # -- public API -------------------------------------------------------
    def resolve(self, name: str, *, country: str = "", create: bool = True) -> Optional[Team]:
        raw = (name or "").strip()
        if not raw:
            return None
        norm = normalise_name(raw)
        if not norm:
            return None

        if norm in self._cache:
            return self.session.get(Team, self._cache[norm])

        canonical = self._curated.get(norm, raw)
        canonical_norm = normalise_name(canonical)

        # 1) exact canonical match
        team = self.session.execute(
            select(Team).where(Team.sport == self.sport, Team.canonical_name == canonical)
        ).scalars().first()

        # 2) previously learned alias (for either spelling)
        if team is None:
            for candidate_norm in (norm, canonical_norm):
                tid = self._alias_lookup(candidate_norm)
                if tid is not None:
                    team = self.session.get(Team, tid)
                    break

        # 3) normalised match against existing canonical names
        if team is None:
            for existing in self._existing_teams():
                if normalise_name(existing.canonical_name) == canonical_norm:
                    team = existing
                    break

        # 4) guarded fuzzy match
        if team is None:
            hit = self._fuzzy(canonical_norm)
            if hit is not None:
                team, score = hit
                self.fuzzy_matches.append((raw, team.canonical_name, round(score, 3)))
                log.info("fuzzy-matched %r -> %r (%.3f)", raw, team.canonical_name, score)

        # 5) create
        if team is None:
            if not create:
                self.unresolved.append(raw)
                return None
            team = Team(
                sport=self.sport,
                canonical_name=canonical,
                short_name=raw if len(raw) <= 24 else "",
                country=country,
            )
            self.session.add(team)
            self.session.flush()
            log.debug("created team %r (from %r)", canonical, raw)

        if team.country == "" and country:
            team.country = country

        self._remember(team, raw, norm)
        if canonical_norm != norm:
            self._remember(team, canonical, canonical_norm)
        self._cache[norm] = team.id
        return team

    def _remember(self, team: Team, alias: str, norm: str) -> None:
        if self._alias_lookup(norm) is not None:
            return
        self.session.add(
            TeamAlias(sport=self.sport, team_id=team.id, alias=alias, alias_norm=norm)
        )
        self.session.flush()

    def report(self) -> dict:
        return {
            "fuzzy_matches": self.fuzzy_matches,
            "unresolved": self.unresolved,
            "cached": len(self._cache),
        }

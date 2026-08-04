"""
Deterministic normal-form labeling from a declared FD set (Phase 2a of TASK_3_PLAN.md).

This module produces the **y** side of the new training data. It is the piece whose
absence makes the current dataset frozen: without it, no newly generated table can be
labeled, so nothing can be regenerated without the label collapsing.

Where it may and may not be used
--------------------------------
The architecture rule from Phase 2::

    Generator --> Iceberg tables (RAW DATA)  -->  nf_features.py  -->  X
         |                                          (sees ONLY the table)
         +------> Manifest (FDs, label, params) -------------------->  y
                      label + validation only

This module is imported by the **generator** (to compute the label) and by the
**validation harness** (2c, to re-check the label against discovered FDs). It must
never be imported by the feature extractor: the extractor sees the materialized table
and nothing else, which makes "copying the label into a feature" structurally
impossible instead of merely forbidden. That is exactly the failure the current
feature set suffers from (findings 1.1-1.3).

What it does not decide
-----------------------
1NF is a property of the *values*, not of the dependencies - no FD set can tell you
whether a cell holds ``"red,green,blue"``. ``violates_1nf`` is therefore measured
outside (by the detector in Phase 0.5) and handed in. It outranks everything else.

Correctness notes (the four places this is usually gotten wrong)
----------------------------------------------------------------
1. **All** candidate keys, not one. Prime-ness is defined over the union of every
   candidate key; taking a single key marks too many attributes non-prime and labels
   3NF tables as 2NF.
2. 2NF is checked via attribute closure, not against the declared FDs literally: a
   partial dependency can be transitive (``A -> C``, ``C -> D`` under key ``{A,B}``).
   Only the *maximal* proper subsets ``K \\ {b}`` need checking - closure is monotone,
   so any smaller proper subset that determines something is dominated by one of them.
3. 3NF has two escape hatches: ``X -> A`` is fine if ``X`` is a superkey **or** ``A``
   is prime. Forgetting the second half labels 3NF tables as 2NF.
4. Attributes appearing only on the right-hand side of FDs are in no candidate key;
   attributes appearing on no right-hand side are in every one. Both facts are used to
   prune the key search, which is otherwise exponential in the column count.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


# A functional dependency X -> Y. Both sides are attribute sets; the right-hand side
# may hold several attributes (``{a, b} -> {c, d}``) and is decomposed where needed.
FunctionalDependency = tuple[frozenset[str], frozenset[str]]

# Guard for the candidate-key search. Only attributes that appear on *both* sides of
# some FD are searched combinatorially (see prune step in ``candidate_keys``); the rest
# is decided up front. A relation that leaves more than this many attributes undecided
# is a modelling error in the generator spec, not a case worth waiting 2**n for.
MAX_SEARCHED_ATTRIBUTES = 20


@dataclass(frozen=True)
class NormalFormAnalysis:
    """Label plus the evidence behind it - consumed by the manifest and by 2c."""

    normal_form: int
    candidate_keys: tuple[frozenset[str], ...]
    prime_attributes: frozenset[str]
    # (determinant, dependent) pairs that break 2NF resp. 3NF. Empty for the form that
    # was reached; the first non-empty one explains the label.
    partial_dependencies: tuple[tuple[frozenset[str], str], ...]
    transitive_dependencies: tuple[tuple[frozenset[str], str], ...]
    reason: str


def normalise_fds(
    fds: Iterable[tuple[Iterable[str], Iterable[str]]],
    attributes: Iterable[str] | None = None,
) -> list[FunctionalDependency]:
    """
    Coerce FDs to ``frozenset`` pairs and drop trivial ones (``X -> A`` with ``A`` in ``X``).

    Args:
        fds: Pairs of iterables of attribute names.
        attributes: If given, every attribute mentioned must occur in it. A typo in a
            generator spec would otherwise produce a silently wrong label - the FD just
            never fires and the table looks better normalized than it is.

    Raises:
        ValueError: An FD mentions an attribute the relation does not have.
    """
    known = frozenset(attributes) if attributes is not None else None
    result: list[FunctionalDependency] = []
    for lhs, rhs in fds:
        left = frozenset(lhs)
        right = frozenset(rhs) - left  # trivial part carries no information
        if known is not None:
            unknown = (left | right) - known
            if unknown:
                message = f"FD mentions unknown attributes {sorted(unknown)}"
                raise ValueError(message)
        if right:
            result.append((left, right))
    return result


def closure(
    attributes: Iterable[str],
    fds: Sequence[FunctionalDependency],
) -> frozenset[str]:
    """
    Attribute closure ``X+``: everything reachable from ``X`` under ``fds``.

    ``X`` is a superkey of ``R`` exactly when ``closure(X) == R``. This is the single
    primitive the rest of the module is built on.
    """
    reachable = set(attributes)
    grew = True
    while grew:
        grew = False
        for lhs, rhs in fds:
            if lhs <= reachable and not rhs <= reachable:
                reachable |= rhs
                grew = True
    return frozenset(reachable)


def candidate_keys(
    attributes: Iterable[str],
    fds: Sequence[FunctionalDependency],
) -> list[frozenset[str]]:
    """
    Every *minimal* superkey, not just one.

    Pruning, both directions of the same observation - an attribute can only be in a key
    if it can determine something, and must be if nothing can determine it:

    * appears on no right-hand side  -> in **every** candidate key (nothing derives it)
    * appears only on right-hand sides -> in **no** candidate key (it derives nothing,
      and is itself derivable from the remaining attributes)

    Only what is left over is searched combinatorially, level by level, skipping any set
    that already contains a key found at a lower level (so what comes back is minimal).

    Raises:
        ValueError: ``attributes`` is empty, or the undecided remainder is too large to
            search (see ``MAX_SEARCHED_ATTRIBUTES``).
    """
    attrs = frozenset(attributes)
    if not attrs:
        message = "A relation needs at least one attribute"
        raise ValueError(message)

    fds = normalise_fds(fds, attrs)
    in_lhs = frozenset().union(*(lhs for lhs, _ in fds)) if fds else frozenset()
    in_rhs = frozenset().union(*(rhs for _, rhs in fds)) if fds else frozenset()

    mandatory = attrs - in_rhs
    searchable = sorted(attrs & in_rhs & in_lhs)

    # Fast path, and the only path when there are no FDs at all: the mandatory part
    # already reaches everything, so it is the one and only candidate key.
    if closure(mandatory, fds) == attrs:
        return [mandatory]

    if len(searchable) > MAX_SEARCHED_ATTRIBUTES:
        message = (
            f"{len(searchable)} attributes appear on both sides of the FD set; the key "
            f"search is exponential in that number (limit {MAX_SEARCHED_ATTRIBUTES}). "
            "Declare the FDs of this table more precisely."
        )
        raise ValueError(message)

    keys: list[frozenset[str]] = []
    for size in range(1, len(searchable) + 1):
        for combo in combinations(searchable, size):
            candidate = mandatory | frozenset(combo)
            # A superset of an already-found key cannot be minimal.
            if any(key <= candidate for key in keys):
                continue
            if closure(candidate, fds) == attrs:
                keys.append(candidate)
    return keys


def prime_attributes(
    attributes: Iterable[str],
    fds: Sequence[FunctionalDependency],
) -> frozenset[str]:
    """Union over **all** candidate keys - the definition 2NF and 3NF are written in."""
    keys = candidate_keys(attributes, fds)
    return frozenset().union(*keys) if keys else frozenset()


def _partial_dependencies(
    keys: Sequence[frozenset[str]],
    fds: Sequence[FunctionalDependency],
    prime: frozenset[str],
) -> list[tuple[frozenset[str], str]]:
    """
    Non-prime attributes reachable from a *proper* subset of a candidate key -> not 2NF.

    Only the maximal proper subsets ``K \\ {b}`` are tested. Closure is monotone, so if
    any proper subset ``X`` of ``K`` reaches ``A``, then some ``K \\ {b}`` with
    ``X <= K \\ {b}`` reaches it too. That turns ``2**|K|`` subsets into ``|K|``.
    """
    found: list[tuple[frozenset[str], str]] = []
    for key in keys:
        for dropped in sorted(key):
            subset = key - {dropped}
            derived = closure(subset, fds) - subset
            for attribute in sorted(derived - prime):
                found.append((subset, attribute))
    return found


def _transitive_dependencies(
    attrs: frozenset[str],
    fds: Sequence[FunctionalDependency],
    prime: frozenset[str],
) -> list[tuple[frozenset[str], str]]:
    """
    FDs that satisfy neither 3NF escape hatch: ``X`` no superkey and ``A`` not prime.

    Checking the declared FDs is enough - if some FD in the closure of the set violates
    3NF, one of the FDs generating it does too.
    """
    found: list[tuple[frozenset[str], str]] = []
    for lhs, rhs in fds:
        if closure(lhs, fds) == attrs:
            continue  # left side is a superkey
        for attribute in sorted(rhs - lhs):
            if attribute not in prime:
                found.append((lhs, attribute))
    return found


def analyze(
    attributes: Iterable[str],
    fds: Iterable[tuple[Iterable[str], Iterable[str]]],
    violates_1nf: bool,
) -> NormalFormAnalysis:
    """
    Full analysis: label plus keys, prime attributes and the violations behind the label.

    Args:
        attributes: All column names of the relation.
        fds: Declared functional dependencies, ``(lhs, rhs)`` pairs of attribute names.
        violates_1nf: Measured outside this module (values are not atomic). Outranks the
            FD analysis: no FD set can express non-atomicity.
    """
    attrs = frozenset(attributes)
    normalised = normalise_fds(fds, attrs)
    keys = candidate_keys(attrs, normalised)
    prime = frozenset().union(*keys) if keys else frozenset()

    partial = _partial_dependencies(keys, normalised, prime)
    transitive = _transitive_dependencies(attrs, normalised, prime)

    if violates_1nf:
        form, reason = 0, "non-atomic values (measured, not derived from the FDs)"
    elif partial:
        determinant, dependent = partial[0]
        form = 1
        reason = (
            f"partial dependency: {{{', '.join(sorted(determinant))}}} -> {dependent}, "
            "a proper subset of a candidate key determines a non-prime attribute"
        )
    elif transitive:
        determinant, dependent = transitive[0]
        form = 2
        reason = (
            f"transitive dependency: {{{', '.join(sorted(determinant))}}} -> {dependent}, "
            "determinant is no superkey and the dependent is not prime"
        )
    else:
        form, reason = 3, "no partial and no transitive dependency"

    return NormalFormAnalysis(
        normal_form=form,
        candidate_keys=tuple(keys),
        prime_attributes=prime,
        partial_dependencies=tuple(partial),
        transitive_dependencies=tuple(transitive),
        reason=reason,
    )


def normal_form(
    attributes: Iterable[str],
    fds: Iterable[tuple[Iterable[str], Iterable[str]]],
    violates_1nf: bool,
) -> int:
    """
    Highest normal form the relation satisfies, 0..3 - the training label.

    ``0`` values are not atomic · ``1`` atomic but partially dependent · ``2`` fully
    dependent but transitively dependent · ``3`` neither.
    """
    return analyze(attributes, fds, violates_1nf).normal_form

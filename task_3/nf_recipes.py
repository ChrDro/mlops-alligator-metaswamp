"""
TPC-H recipes for the normal-form training set (Phase 2b of TASK_3_PLAN.md).

The FD algebra lives here, the label computation in ``nf_labeling``, the writing in
``nf_generator``. Nothing in this module talks to a database.

Where the dependencies come from
--------------------------------
Not from the TPC-H specification and not from memory - they were measured. An exhaustive
single-attribute FD scan over ``tpch.tiny`` produced candidates, each re-checked against
``tpch.sf1`` to separate real dependencies from small-sample artifacts:

| candidate                        | verdict  | note                                     |
| :------------------------------- | :------- | :--------------------------------------- |
| `part.brand -> mfgr`             | real     | brand encodes the manufacturer           |
| `lineitem.shipdate -> linestatus`| real     | linestatus is a function of the ship date|
| `customer.acctbal -> mktsegment` | artifact | 1499 distinct values over 1500 rows      |
| `orders.totalprice -> status`    | artifact | same reason, breaks at sf1               |

Two consequences the recipes are built around: ``lineitem`` on its own is **2NF**, not
3NF, and ``part`` with ``mfgr`` is 2NF as well. Each is only 3NF once the dependent
attribute is projected away - which is where the 3NF-with-composite-key tables come from.

Two columns that were excluded and are now back on purpose
----------------------------------------------------------
``orders.shippriority`` (constant, one value over 15.000 rows) and ``customer.acctbal``
(1499 distinct over 1500) were removed from this catalog because they produced accidental
dependencies that polluted the labels. That was the wrong fix, and a prediction run on real
TPC-H showed why: `orders` came back 2NF because every column "determines" a constant one,
and `customer` came back 2NF from ``acctbal -> mktsegment``. Both are 3NF.

The guards belong in the extractor, not in the choice of test data - a trivial dependency is
trivial wherever it appears, and ``find_dependencies`` now drops constant dependents and
near-unique determinants. Removing the columns instead meant the training set contained
neither phenomenon, so nothing could learn them and no check could catch the regression. They
are back as the decoys they always were: if a guard breaks, the label-versus-discovered check
fails on these tables instead of on a customer's database.

What is deliberately not projected
----------------------------------
* ``customer/supplier.name|address|phone`` and ``part.name`` - accidentally unique. They
  would be extra candidate keys, and ``part.name`` is unique on ``tiny`` but *not* on
  ``sf1``, so the label would depend on the scale factor. ``nation.name`` and
  ``region.name`` are kept: stable natural keys at every scale, and the only thing
  keeping natural keys represented at all.
* ``supplier.acctbal`` and ``lineitem.extendedprice`` - near-unique, 1499 resp. 1968
  distinct values at 1500-2000 rows. ``customer.acctbal`` is projected as the deliberate
  near-unique decoy (see above); keeping the other two out limits how many tables one guard
  regression can move, so a failure points somewhere specific.
* ``comment`` - except where a recipe wants free text on purpose (see ``DECOY_RECIPES``).

The two things this set is built to defeat
------------------------------------------
1. **Composite key must not predict the label.** 1NF implies a composite key by
   definition, so the converse has to be broken on purpose: ``lineitem`` and ``partsupp``
   supply 3NF *and* 2NF tables with composite keys.
2. **0NF must not be distinguishable by anything but atomicity.** Every 0NF table is
   generated together with a *matched control*: identical query, identical grain,
   identical row count, identical column count, with the list column replaced by an
   atomic aggregate. The pair differs in exactly one property - the one the atomicity
   features have to pick up. This is the direct answer to finding 1.3, where 0NF and 3NF
   were separated by a generator artifact instead of by the data.

Table names carry no hint of the label (``nf_017_tiny_real``). The extractor sees table
and column names, so a name like ``tbl_dirty_1nf_7`` would be a leak it could read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from nf_generator import TableSpec
from nf_labeling import FunctionalDependency, closure, normalise_fds


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


# --------------------------------------------------------------------------------------
# Base relations - attributes, keys and FDs as measured, not as assumed
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BaseRelation:
    """
    One TPC-H table, reduced to what the recipes may project.

    Attributes:
        name: Table name in the ``tpch`` catalog.
        key: Primary key. Determines every other attribute.
        attributes: Everything a recipe may select, key included.
        alternate_keys: Further verified keys (``nation.name``). Each determines the rest.
        extra_fds: Verified dependencies that are neither key nor alternate key.
    """

    name: str
    key: frozenset[str]
    attributes: tuple[str, ...]
    alternate_keys: tuple[frozenset[str], ...] = ()
    extra_fds: tuple[FunctionalDependency, ...] = ()

    def fds(self) -> list[FunctionalDependency]:
        """Every dependency that holds inside this relation."""
        everything = frozenset(self.attributes)
        result = [(self.key, everything - self.key)]
        result += [(alternate, everything - alternate) for alternate in self.alternate_keys]
        result += list(self.extra_fds)
        return normalise_fds(result, self.attributes)


TPCH: dict[str, BaseRelation] = {
    "region": BaseRelation(
        name="region",
        key=frozenset({"regionkey"}),
        attributes=("regionkey", "name"),
        alternate_keys=(frozenset({"name"}),),
    ),
    "nation": BaseRelation(
        name="nation",
        key=frozenset({"nationkey"}),
        attributes=("nationkey", "name", "regionkey"),
        alternate_keys=(frozenset({"name"}),),
    ),
    "supplier": BaseRelation(
        name="supplier",
        key=frozenset({"suppkey"}),
        attributes=("suppkey", "nationkey"),
    ),
    "customer": BaseRelation(
        name="customer",
        # acctbal is the near-unique decoy: 1499 distinct over 1500 rows, which is close
        # enough to a key that it determines other columns by accident. No declared FD - it
        # is a plain non-key attribute, and any dependency out of it is the artifact
        # find_dependencies is meant to drop.
        key=frozenset({"custkey"}),
        attributes=("custkey", "nationkey", "mktsegment", "acctbal"),
    ),
    "part": BaseRelation(
        name="part",
        key=frozenset({"partkey"}),
        attributes=("partkey", "mfgr", "brand", "type", "size", "container", "retailprice"),
        # Verified on sf1: brand is 'Brand#' || manufacturer digit || brand digit.
        extra_fds=((frozenset({"brand"}), frozenset({"mfgr"})),),
    ),
    "partsupp": BaseRelation(
        name="partsupp",
        key=frozenset({"partkey", "suppkey"}),
        attributes=("partkey", "suppkey", "availqty", "supplycost"),
    ),
    "orders": BaseRelation(
        name="orders",
        # shippriority is the constant decoy: 0 in every TPC-H row. Every other column
        # "determines" it, which is a dependency on the empty set and says nothing about the
        # schema. Declared like any other non-key attribute - `orderkey -> shippriority`
        # holds, and that is the whole truth about it.
        key=frozenset({"orderkey"}),
        attributes=(
            "orderkey",
            "custkey",
            "orderstatus",
            "orderdate",
            "orderpriority",
            "clerk",
            "shippriority",
        ),
    ),
    "lineitem": BaseRelation(
        name="lineitem",
        key=frozenset({"orderkey", "linenumber"}),
        attributes=(
            "orderkey",
            "linenumber",
            "partkey",
            "suppkey",
            "quantity",
            "discount",
            "returnflag",
            "linestatus",
            "shipdate",
            "shipmode",
        ),
        # Verified on sf1: linestatus is 'O' after a cutoff ship date, 'F' before it.
        extra_fds=((frozenset({"shipdate"}), frozenset({"linestatus"})),),
    ),
}


# --------------------------------------------------------------------------------------
# FD algebra
# --------------------------------------------------------------------------------------


def project_fds(
    fds: Sequence[FunctionalDependency],
    keep: Iterable[str],
) -> list[FunctionalDependency]:
    """
    The dependencies that survive dropping columns.

    Filtering the FD list is **not** enough. With ``A -> B`` and ``B -> C``, dropping
    ``B`` leaves ``A -> C`` standing - discarding both FDs because they mention ``B``
    would make the projection look better normalized than it is, which is the direction
    of error nothing downstream would question.

    So every determinant is re-closed and intersected with what is kept. Candidate
    determinants are the surviving left-hand sides plus all single attributes; for the FD
    shapes used here (key -> rest, single -> single) that is a complete cover, and every
    dependency it produces provably holds, because it comes out of a closure.
    """
    kept = frozenset(keep)
    determinants = {lhs for lhs, _ in fds if lhs <= kept}
    determinants |= {frozenset({attribute}) for attribute in kept}

    projected = [(lhs, (closure(lhs, fds) & kept) - lhs) for lhs in determinants]
    return normalise_fds([(lhs, rhs) for lhs, rhs in projected if rhs], kept)


def rename_fds(
    fds: Sequence[FunctionalDependency],
    mapping: dict[str, str],
) -> list[FunctionalDependency]:
    """Apply an attribute rename to both sides of every FD."""
    return [
        (frozenset(mapping.get(a, a) for a in lhs), frozenset(mapping.get(a, a) for a in rhs))
        for lhs, rhs in fds
    ]


# --------------------------------------------------------------------------------------
# Joins
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class JoinStep:
    """
    One denormalization step: pull a dimension in over a foreign key.

    Attributes:
        fk: Column in the attribute set built so far that holds the dimension's key.
        dim: Name of the dimension in ``TPCH``. Its key column is not projected again -
            the foreign key already carries those values.
    """

    fk: str
    dim: str


@dataclass(frozen=True)
class Relation:
    """A built relation: what it contains, what holds in it, and the SQL that makes it."""

    attributes: tuple[str, ...]
    fds: tuple[FunctionalDependency, ...]
    sql: str


def build_join(
    fact: str,
    steps: Sequence[JoinStep] = (),
    drop: Iterable[str] = (),
    extra_select: Sequence[str] = (),
    extra_attributes: Sequence[str] = (),
    extra_fds: Sequence[FunctionalDependency] = (),
    source_schema: str = "tiny",
    limit: int | None = None,
) -> Relation:
    """
    Fold dimensions into a fact table and derive the resulting dependencies.

    The rule, applied once per step: ``FDs(fact) u FDs(dim) u {fk -> dim attributes}``.
    The dimension's key column is not projected again - the foreign key already holds
    those values, so the dimension's own dependencies are rewritten onto it. An alternate
    key inside the dimension (``nation.name -> nationkey``) therefore survives the join.

    Candidate keys are never stated anywhere; they follow from the FD set and are computed
    by ``nf_labeling``. That is deliberate - a join ladder whose key was guessed is how
    the current feature set ended up guessing (finding 1.4).

    Args:
        fact: Name of the fact relation in ``TPCH``.
        steps: Dimensions to fold in, in order. Each ``fk`` must already exist.
        drop: Attributes to project away afterwards.
        extra_select: Additional select expressions, e.g. the free-text decoy column.
        extra_attributes: Names of those additional columns. They are determined by the
            fact key like every other non-key attribute.
        extra_fds: Dependencies that hold among the columns beyond what the joins imply -
            the pair-determinant recipes declare ``{a, b} -> derived`` here.
        source_schema: Schema inside the ``tpch`` catalog ("tiny", "sf1", ...).
        limit: Row cap on the fact table, applied before the joins and ordered by the
            fact key - an unordered LIMIT in Trino does not reproduce.
    """
    fact_relation = TPCH[fact]
    attributes = list(fact_relation.attributes)
    fds = fact_relation.fds()

    fact_sql = f"tpch.{source_schema}.{fact}"
    if limit is not None:
        # Declaration order, NOT sorted order. Sorting the key alphabetically puts
        # `linenumber` before `orderkey` for lineitem, and the first 2000 rows of that
        # ordering all carry linenumber = 1: the column goes constant, orderkey becomes
        # unique, and the composite key the recipe is built around disappears from the
        # sample. The sample has to preserve the key structure, not just the row count.
        key_order = ", ".join(
            f'"{column}"' for column in fact_relation.attributes if column in fact_relation.key
        )
        # Identifiers come from the TPCH catalog above, not from anything external.
        fact_sql = f"(SELECT * FROM {fact_sql} ORDER BY {key_order} LIMIT {limit})"  # noqa: S608

    select_list = [f't0."{attribute}" AS "{attribute}"' for attribute in fact_relation.attributes]
    from_clause = f"{fact_sql} t0"
    # Which SQL alias exposes an attribute, and under which name there - this is what
    # makes chained dimensions (orders -> customer -> nation) join to the right alias.
    origin: dict[str, tuple[str, str]] = {a: ("t0", a) for a in fact_relation.attributes}

    for index, step in enumerate(steps, start=1):
        if step.fk not in origin:
            message = f"join step {step.dim} needs {step.fk!r}, which is not in the relation yet"
            raise ValueError(message)

        dim = TPCH[step.dim]
        if len(dim.key) != 1:
            message = f"{step.dim} has a composite key; only single-column joins are supported"
            raise ValueError(message)
        dim_key = next(iter(dim.key))
        alias = f"t{index}"

        source_alias, source_column = origin[step.fk]
        from_clause += (
            f"\n  JOIN tpch.{source_schema}.{step.dim} {alias}"
            f' ON {source_alias}."{source_column}" = {alias}."{dim_key}"'
        )

        # The dimension's key column maps onto the foreign key; everything else gets a
        # prefix so two dimensions can never collide on `name`.
        mapping = {a: f"{step.dim}_{a}" for a in dim.attributes}
        mapping[dim_key] = step.fk
        for attribute in dim.attributes:
            if attribute == dim_key:
                continue
            new_name = mapping[attribute]
            attributes.append(new_name)
            select_list.append(f'{alias}."{attribute}" AS "{new_name}"')
            origin[new_name] = (alias, attribute)

        fds += rename_fds(dim.fds(), mapping)

    for name, expression in zip(extra_attributes, extra_select, strict=True):
        attributes.append(name)
        select_list.append(f'{expression} AS "{name}"')
        fds.append((fact_relation.key, frozenset({name})))
    # Beyond key -> column: a derived column is also a function of the columns it was
    # computed from, and that is the dependency the pair recipes exist to declare.
    fds += list(extra_fds)

    keep = [attribute for attribute in attributes if attribute not in set(drop)]
    select_list = [
        expression
        for attribute, expression in zip(attributes, select_list, strict=True)
        if attribute in set(keep)
    ]
    sql = "SELECT " + ",\n       ".join(select_list) + f"\nFROM {from_clause}"
    return Relation(
        attributes=tuple(keep),
        fds=tuple(project_fds(normalise_fds(fds, attributes), keep)),
        sql=sql,
    )


# --------------------------------------------------------------------------------------
# Recipes
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class JoinRecipe:
    """
    A join-ladder shape, before source schema and naming variant are applied.

    Attributes:
        recipe_id: Stable identifier. Goes into ``generation_params``, never into the
            table name.
        fact: Fact relation.
        steps: Dimensions to fold in.
        drop: Attributes to project away after the join.
        decoy: Project the TPC-H ``comment`` column as free text.
        derived: Extra columns computed from fact columns, as ``(name, sql expression
            over t0, determinant columns)``. Each declares ``{determinants} -> name`` -
            the composite-determinant dependency no single-column search can see
            (the Willibald gap, 2026-08-07).
        note: What this shape is for. Travels into the manifest.
    """

    recipe_id: str
    fact: str
    steps: tuple[JoinStep, ...] = ()
    drop: frozenset[str] = frozenset()
    decoy: bool = False
    derived: tuple[tuple[str, str, tuple[str, ...]], ...] = ()
    note: str = ""

    def build(self, source_schema: str, limit: int) -> Relation:
        extra_select = ['t0."comment"'] if self.decoy else []
        extra_attributes = ["note_text"] if self.decoy else []
        extra_fds = []
        for name, expression, determinants in self.derived:
            extra_select.append(expression)
            extra_attributes.append(name)
            extra_fds.append((frozenset(determinants), frozenset({name})))
        return build_join(
            fact=self.fact,
            steps=self.steps,
            drop=self.drop,
            extra_select=tuple(extra_select),
            extra_attributes=tuple(extra_attributes),
            extra_fds=tuple(extra_fds),
            source_schema=source_schema,
            limit=limit,
        )


JOIN_RECIPES: tuple[JoinRecipe, ...] = (
    # ---------------------------------------------------------------- 3NF, single key
    JoinRecipe("orders_core", "orders", decoy=True, note="3NF, surrogate single key"),
    JoinRecipe("customer_core", "customer", note="3NF, low-cardinality attributes"),
    JoinRecipe("supplier_core", "supplier", note="3NF, single key"),
    JoinRecipe(
        "part_core",
        "part",
        drop=frozenset({"mfgr"}),
        note="3NF only because mfgr is dropped - brand -> mfgr would make it 2NF",
    ),
    JoinRecipe("nation_core", "nation", note="3NF, natural alternate key (name)"),
    # ------------------------------------------------------------- 3NF, composite key
    # Here so that "composite key" does not predict the label: 1NF implies a composite
    # key, and without these the converse would be learnable too.
    JoinRecipe(
        "lineitem_core",
        "lineitem",
        drop=frozenset({"linestatus"}),
        decoy=True,
        note="3NF with a composite key - anti-leak against table_has_composite_pk",
    ),
    JoinRecipe("partsupp_core", "partsupp", note="3NF with a composite key"),
    # ---------------------------------------------------------------- 2NF, single key
    JoinRecipe(
        "orders_customer",
        "orders",
        steps=(JoinStep("custkey", "customer"),),
        decoy=True,
        note="2NF: custkey -> customer attributes, transitive over the key",
    ),
    JoinRecipe(
        "orders_customer_nation",
        "orders",
        steps=(JoinStep("custkey", "customer"), JoinStep("customer_nationkey", "nation")),
        note="2NF, two-step transitive chain",
    ),
    JoinRecipe(
        "orders_customer_nation_region",
        "orders",
        steps=(
            JoinStep("custkey", "customer"),
            JoinStep("customer_nationkey", "nation"),
            JoinStep("nation_regionkey", "region"),
        ),
        note="2NF, three-step transitive chain",
    ),
    JoinRecipe(
        "customer_nation",
        "customer",
        steps=(JoinStep("nationkey", "nation"),),
        note="2NF, single join level",
    ),
    JoinRecipe(
        "supplier_nation",
        "supplier",
        steps=(JoinStep("nationkey", "nation"),),
        note="2NF, single join level",
    ),
    JoinRecipe(
        "part_with_mfgr",
        "part",
        note="2NF without any join - brand -> mfgr is a real TPC-H dependency",
    ),
    # ------------------------------------------------------------- 2NF, composite key
    JoinRecipe(
        "lineitem_with_status",
        "lineitem",
        decoy=True,
        note="2NF with a composite key: shipdate -> linestatus, neither is the key",
    ),
    JoinRecipe(
        "lineitem_part",
        "lineitem",
        steps=(JoinStep("partkey", "part"),),
        drop=frozenset({"linestatus"}),
        note="2NF: partkey is not part of the key, so the dependency is transitive",
    ),
    JoinRecipe(
        "lineitem_supplier_nation",
        "lineitem",
        steps=(JoinStep("suppkey", "supplier"), JoinStep("supplier_nationkey", "nation")),
        drop=frozenset({"linestatus"}),
        note="2NF with a composite key, two join levels",
    ),
    # ---------------------------------------------------- 2NF via a PAIR determinant
    # The Willibald gap (2026-08-07): {KatID, Umfang} -> Typ, {Bestelldatum, Wunschdatum}
    # -> Rabatt. A violation whose determinant is a column pair was invisible to the
    # single-attribute FD search AND absent from this training set, so the model read
    # every such table as clean 3NF at 0.97+ confidence. The derived column is a function
    # of two low-cardinality fact columns - neither alone determines it.
    JoinRecipe(
        "lineitem_ship_class",
        "lineitem",
        drop=frozenset({"linestatus"}),
        derived=(
            (
                "ship_class",
                'substr(t0."shipmode", 1, 2) || t0."returnflag"',
                ("shipmode", "returnflag"),
            ),
        ),
        note="2NF via a pair determinant: {shipmode, returnflag} -> ship_class, "
        "on a composite-key fact",
    ),
    JoinRecipe(
        "orders_handling_code",
        "orders",
        derived=(
            (
                "handling_code",
                'substr(t0."orderpriority", 1, 1) || t0."orderstatus"',
                ("orderpriority", "orderstatus"),
            ),
        ),
        note="2NF via a pair determinant: {orderpriority, orderstatus} -> handling_code, "
        "on a single-key fact",
    ),
    # ------------------------------------------------------- 1NF, partial dependencies
    JoinRecipe(
        "lineitem_orders",
        "lineitem",
        steps=(JoinStep("orderkey", "orders"),),
        drop=frozenset({"linestatus"}),
        decoy=True,
        note="1NF: orderkey is half the key and drags the order attributes along",
    ),
    JoinRecipe(
        "lineitem_orders_customer",
        "lineitem",
        steps=(JoinStep("orderkey", "orders"), JoinStep("orders_custkey", "customer")),
        drop=frozenset({"linestatus"}),
        note="1NF: partial and transitive at once - 2NF is decided first",
    ),
    JoinRecipe(
        "partsupp_supplier",
        "partsupp",
        steps=(JoinStep("suppkey", "supplier"),),
        decoy=True,
        note="1NF: suppkey is half the key",
    ),
    JoinRecipe(
        "partsupp_part",
        "partsupp",
        steps=(JoinStep("partkey", "part"),),
        note="1NF: partkey is half the key",
    ),
    JoinRecipe(
        "partsupp_part_supplier",
        "partsupp",
        steps=(JoinStep("partkey", "part"), JoinStep("suppkey", "supplier")),
        note="1NF: both halves of the key drag a dimension along",
    ),
)


# --------------------------------------------------------------------------------------
# 0NF by injection, each with its matched control
# --------------------------------------------------------------------------------------

# Parents with only one child row would produce a single value, which is not list-like -
# the detector would rightly stay quiet and the label would be noise. Requiring at least
# two children makes the 0NF label honest. The sparse case (a `tags` column where most
# rows hold one tag) is a threshold question for Phase 0.5, not a labeling question.
MIN_CHILDREN = 2


@dataclass(frozen=True)
class ListRecipe:
    """
    Fold a 1:n child into one list-valued column - and build the matched control.

    Attributes:
        recipe_id: Stable identifier.
        parent: Join recipe supplying the parent grain.
        child: Child relation in ``TPCH``.
        child_fk: Column in the child that points at the parent.
        parent_fk: The parent column it points at.
        value: Child column folded into the list. Deliberately low-cardinality: a list of
            shipping modes is what a denormalized column actually looks like, and a
            near-unique one (a price, a key) would turn the control into an accidental
            key of its own.
        child_order: Child column deciding which value the control keeps. Has to vary
            inside a parent group, or the control column goes constant.
        separator: ``,``, ``;`` or ``|``.
        note: Travels into the manifest.
    """

    recipe_id: str
    parent: str
    child: str
    child_fk: str
    parent_fk: str
    value: str
    child_order: str
    separator: str
    note: str = ""

    def build(self, source_schema: str, limit: int, *, atomic: bool) -> Relation:
        """
        Args:
            atomic: ``False`` builds the 0NF table, ``True`` its matched control. Both
                share query, grain, row count and column count - only the last column
                differs, and with it the atomicity.
        """
        parent = JOIN_RECIPES_BY_ID[self.parent].build(source_schema, limit)
        group_by = ",\n         ".join(f'p."{attribute}"' for attribute in parent.attributes)
        select = ",\n       ".join(
            f'p."{attribute}" AS "{attribute}"' for attribute in parent.attributes
        )

        if atomic:
            # One child value instead of all of them - the tightest possible contrast to
            # the list: same source column, same type, one entry rather than many.
            #
            # NOT count(*), which is what this was first. TPC-H has fixed fan-outs (every
            # part has exactly four suppliers), so the count came out **constant** - and a
            # constant column is determined by every other column, which dragged the
            # control down to 2NF. The validation harness (2c) caught it on six tables.
            column = f"{self.child}_{self.value}_first"
            aggregate = f'CAST(min_by(c."{self.value}", c."{self.child_order}") AS VARCHAR)'
        else:
            column = f"{self.child}_{self.value}_list"
            aggregate = (
                f'array_join(array_agg(CAST(c."{self.value}" AS VARCHAR) '
                f"ORDER BY c.\"{self.value}\"), '{self.separator}')"
            )

        sql = (
            f'SELECT {select},\n       {aggregate} AS "{column}"\n'
            f"FROM (\n{_indent(parent.sql)}\n) p\n"
            f"  JOIN tpch.{source_schema}.{self.child} c"
            f' ON p."{self.parent_fk}" = c."{self.child_fk}"\n'
            f"GROUP BY {group_by}\n"
            f"HAVING count(*) >= {MIN_CHILDREN}"
        )

        attributes = (*parent.attributes, column)
        # Aggregating to the parent grain keeps the parent's dependencies and adds one:
        # the new column is determined by whatever determines the parent row.
        fds = [*parent.fds, (frozenset({self.parent_fk}), frozenset({column}))]
        return Relation(attributes=attributes, fds=tuple(normalise_fds(fds, attributes)), sql=sql)


LIST_RECIPES: tuple[ListRecipe, ...] = (
    ListRecipe(
        "orders_shipmode_list",
        parent="orders_core",
        child="lineitem",
        child_fk="orderkey",
        parent_fk="orderkey",
        value="shipmode",
        child_order="linenumber",
        separator=",",
        note="0NF on a 3NF parent - the control keeps one shipping mode instead of all",
    ),
    ListRecipe(
        "customer_orderpriority_list",
        parent="customer_core",
        child="orders",
        child_fk="custkey",
        parent_fk="custkey",
        value="orderpriority",
        child_order="orderkey",
        separator=";",
        note="0NF with a semicolon separator",
    ),
    ListRecipe(
        "supplier_shipmode_list",
        parent="supplier_core",
        child="lineitem",
        child_fk="suppkey",
        parent_fk="suppkey",
        value="shipmode",
        child_order="orderkey",
        separator="|",
        note="0NF with a pipe separator, and a long list - the case that showed the "
        "prose guard has to measure per token, not per value",
    ),
    ListRecipe(
        "part_returnflag_list",
        parent="part_core",
        child="lineitem",
        child_fk="partkey",
        parent_fk="partkey",
        value="returnflag",
        child_order="orderkey",
        separator=",",
        note="0NF on a 3NF parent, second shape",
    ),
    ListRecipe(
        "orders_customer_shipmode_list",
        parent="orders_customer",
        child="lineitem",
        child_fk="orderkey",
        parent_fk="orderkey",
        value="shipmode",
        child_order="linenumber",
        separator=",",
        note="0NF on a 2NF parent - 0NF must not correlate with the parent's form",
    ),
)


@dataclass(frozen=True)
class RepeatingGroupRecipe:
    """
    The other 1NF violation: ``order_1``, ``order_2``, ``order_3``.

    No separator anywhere, so the value-based detector cannot see this one - it is
    visible only in the column *names*. That is on purpose: it is what forces a
    name-pattern feature to exist, and it is why the atomicity family cannot consist of
    value statistics alone.
    """

    recipe_id: str
    parent: str
    child: str
    child_fk: str
    parent_fk: str
    value: str
    width: int = 3
    note: str = ""

    def build(self, source_schema: str, limit: int, *, atomic: bool) -> Relation:
        parent = JOIN_RECIPES_BY_ID[self.parent].build(source_schema, limit)
        group_by = ",\n         ".join(f'"{attribute}"' for attribute in parent.attributes)
        select = ",\n       ".join(f'"{attribute}"' for attribute in parent.attributes)

        if atomic:
            # Same column count and grain, no repeating group: one ranked value plus two
            # ordinary aggregates.
            columns = [f"first_{self.value}", "child_count", f"last_{self.value}"]
            aggregates = [
                f'max(CASE WHEN rn = 1 THEN "{self.value}" END)',
                "count(*)",
                f'max("{self.value}")',
            ]
        else:
            columns = [f"{self.value}_{i}" for i in range(1, self.width + 1)]
            aggregates = [
                f'max(CASE WHEN rn = {i} THEN "{self.value}" END)' for i in range(1, self.width + 1)
            ]

        projection = ",\n       ".join(
            f'{aggregate} AS "{column}"'
            for column, aggregate in zip(columns, aggregates, strict=True)
        )
        sql = (
            f"SELECT {select},\n       {projection}\n"  # noqa: S608 - TPCH identifiers
            f"FROM (\n"
            f'  SELECT p.*, c."{self.value}",\n'
            f'         row_number() OVER (PARTITION BY p."{self.parent_fk}"'
            f' ORDER BY c."{self.value}") AS rn\n'
            f"  FROM (\n{_indent(parent.sql, 4)}\n  ) p\n"
            f"    JOIN tpch.{source_schema}.{self.child} c"
            f' ON p."{self.parent_fk}" = c."{self.child_fk}"\n'
            f")\n"
            f"WHERE rn <= {self.width}\n"
            f"GROUP BY {group_by}\n"
            f"HAVING count(*) >= {MIN_CHILDREN}"
        )

        attributes = (*parent.attributes, *columns)
        fds = [*parent.fds]
        fds += [(frozenset({self.parent_fk}), frozenset({column})) for column in columns]
        return Relation(attributes=attributes, fds=tuple(normalise_fds(fds, attributes)), sql=sql)


REPEATING_RECIPES: tuple[RepeatingGroupRecipe, ...] = (
    RepeatingGroupRecipe(
        "customer_order_columns",
        parent="customer_core",
        child="orders",
        child_fk="custkey",
        parent_fk="custkey",
        value="orderkey",
        note="0NF as a repeating group - invisible to any value-based detector",
    ),
)


JOIN_RECIPES_BY_ID = {recipe.recipe_id: recipe for recipe in JOIN_RECIPES}


def _indent(sql: str, spaces: int = 2) -> str:
    pad = " " * spaces
    return "\n".join(pad + line for line in sql.splitlines())


# --------------------------------------------------------------------------------------
# Variation axes
# --------------------------------------------------------------------------------------

# (schema in the tpch catalog, row cap on the fact table). The third tier is small on
# purpose: real prediction targets (Willibald period 1) run 6 to 2000 rows, and a model
# trained only on 2000+ rows meets near-unique columns and guard-dropped dependencies it
# has never seen at that scale.
SOURCES: tuple[tuple[str, int], ...] = (("tiny", 2000), ("sf1", 20000), ("tiny", 150))

# Obfuscated names are not cosmetic. The current model leans on name features
# (`name_ends_with_id`, finding 1.2); half the tables carrying meaningless column names
# is what stops that from working.
NAMINGS: tuple[str, ...] = ("real", "obfuscated")


def obfuscate(attributes: Sequence[str]) -> dict[str, str]:
    """
    Positional names without digits: ``xa``, ``xb``, ..., ``xz``, ``xaa``.

    Digits are avoided on purpose. A numeric suffix (``col_01``, ``col_02``) is exactly
    the pattern the repeating-group detector looks for, so numbered obfuscation would
    turn every obfuscated table into a fake 1NF violation.
    """
    return {attribute: f"x{_letters(i)}" for i, attribute in enumerate(attributes)}


def _letters(index: int) -> str:
    """0 -> 'a', 25 -> 'z', 26 -> 'aa'."""
    name = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(ord("a") + remainder) + name
    return name


def apply_naming(relation: Relation, naming: str) -> Relation:
    """Rename in an outer SELECT so the builders stay unaware of the naming axis."""
    if naming == "real":
        return relation

    mapping = obfuscate(relation.attributes)
    projection = ",\n       ".join(
        f'"{attribute}" AS "{mapping[attribute]}"' for attribute in relation.attributes
    )
    return Relation(
        attributes=tuple(mapping[attribute] for attribute in relation.attributes),
        fds=tuple(rename_fds(relation.fds, mapping)),
        sql=f"SELECT {projection}\nFROM (\n{_indent(relation.sql)}\n)",
    )


@dataclass(frozen=True)
class _Variant:
    """One (recipe, source, naming) combination, before it becomes a TableSpec."""

    recipe_id: str
    kind: str
    relation: Relation
    violates_1nf: bool
    params: dict[str, object] = field(default_factory=dict)


def _variants() -> list[_Variant]:
    """Every table this module knows how to build, in a stable order."""
    variants: list[_Variant] = []
    for source_schema, limit in SOURCES:
        for naming in NAMINGS:
            common = {"source_schema": source_schema, "row_limit": limit, "naming": naming}

            for recipe in JOIN_RECIPES:
                relation = apply_naming(recipe.build(source_schema, limit), naming)
                variants.append(
                    _Variant(
                        recipe_id=recipe.recipe_id,
                        kind="join",
                        relation=relation,
                        violates_1nf=False,
                        params={**common, "note": recipe.note, "has_decoy": recipe.decoy},
                    ),
                )

            for injection in (*LIST_RECIPES, *REPEATING_RECIPES):
                # The 0NF table and its control are emitted next to each other and linked
                # by pair_id, so any evaluation can be run on the pairs alone.
                for atomic in (False, True):
                    relation = apply_naming(
                        injection.build(source_schema, limit, atomic=atomic),
                        naming,
                    )
                    variants.append(
                        _Variant(
                            recipe_id=f"{injection.recipe_id}{'_control' if atomic else ''}",
                            kind="control" if atomic else "injection",
                            relation=relation,
                            violates_1nf=not atomic,
                            params={
                                **common,
                                "note": injection.note,
                                # The limit is part of the identity: two tiers share the
                                # "tiny" schema now, and a pair id that ignored the limit
                                # would glue four tables into two pairs.
                                "pair_id": (
                                    f"{injection.recipe_id}_{source_schema}_{limit}_{naming}"
                                ),
                            },
                        ),
                    )
    return variants


def build_specs() -> list[TableSpec]:
    """
    Every recipe crossed with every variation axis, as generator specs.

    Table names are positional and carry no hint of the label - the extractor sees them.
    The recipe behind a table is recorded in ``generation_params`` instead.
    """
    specs = []
    for index, variant in enumerate(_variants(), start=1):
        source = variant.params["source_schema"]
        naming = "real" if variant.params["naming"] == "real" else "obf"
        specs.append(
            TableSpec(
                table_name=f"nf_{index:03d}_{source}_{naming}",
                source_sql=variant.relation.sql,
                attributes=variant.relation.attributes,
                declared_fds=variant.relation.fds,
                violates_1nf=variant.violates_1nf,
                generation_params={
                    "recipe_id": variant.recipe_id,
                    "family": variant.recipe_id,
                    "source": "tpch",
                    "kind": variant.kind,
                    **variant.params,
                },
            ),
        )
    return specs

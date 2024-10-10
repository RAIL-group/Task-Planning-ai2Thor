def get_domain(whole_graph):
    loc_set = set()
    for c_idx in whole_graph['cnt_node_idx']:
        loc_set.add(whole_graph['node_names'][c_idx])
    loc_str = ''
    for loc in loc_set:
        loc_str += ' ' + loc

    obj_set = set()
    for o_idx in whole_graph['obj_node_idx']:
        obj_set.add(whole_graph['node_names'][o_idx])
    obj_str = ''
    for obj in obj_set:
        obj_str += ' ' + obj

    DOMAIN_PDDL = f"""
    (define
    (domain indoor)

    (:requirements :strips :typing :action-costs :existential-preconditions)

    (:types
        location item - object
        init_r{loc_str} - location
        {obj_str} - item
    )

    (:predicates
        (is-holding ?obj - item)
        (is-located ?obj - item)
        (is-at ?obj - item ?loc - location)
        (rob-at ?loc - location)
        (hand-is-free)
        (filled-with ?obj - item ?cnt - item)
        (is-liquid ?obj - item)
        (is-pickable ?obj - item)
        (is-fillable ?obj - item)
        (restrict-move-to ?loc - location)
        (spread-applied ?obj1 - item ?obj2 - spread)
        (is-spread ?obj - item)
        (is-spreadable ?obj - item)
        (is-washable ?obj - item)
        (is-dirty ?obj - item)
        (is-folded ?obj - item)
        (is-foldable ?obj - item)
        (ban-move)
    )

    (:functions
        (known-cost ?start ?end)
        (find-cost ?obj ?loc)
        (total-cost)
    )

    (:action fold
        :parameters (?npkn - napkin)
        :precondition (and
            (hand-is-free)
            (not (is-folded ?npkn))
            (is-foldable ?npkn)
            (is-at ?npkn countertop)
            (rob-at countertop)
        )
        :effect (and
            (is-folded ?npkn)
            (increase (total-cost) 50)
        )
    )

    (:action apply-spread
        :parameters (?s - spread ?k - knife)
        :precondition (and
            (rob-at countertop)
            (is-at bread countertop)
            (is-at ?s countertop)
            (is-holding ?k)
            (not (is-dirty ?k))
            (is-spread ?s)
            (is-spreadable bread)
            (not (spread-applied bread ?s))
        )
        :effect (and
            (spread-applied bread ?s)
            (is-dirty ?k)
        )
    )

    (:action pick
        :parameters (?obj - item ?loc - location)
        :precondition (and
            (is-pickable ?obj)
            (is-located ?obj)
            (is-at ?obj ?loc)
            (rob-at ?loc)
            (hand-is-free)
        )
        :effect (and
            (not (is-at ?obj ?loc))
            (is-holding ?obj)
            (not (hand-is-free))
            (not (ban-move))
            (increase (total-cost) 100)
        )
    )

    (:action place
        :parameters (?obj - item ?loc - location)
        :precondition (and
            (not (hand-is-free))
            (rob-at ?loc)
            (is-holding ?obj)
        )
        :effect (and
            (is-at ?obj ?loc)
            (not (is-holding ?obj))
            (hand-is-free)
            (not (ban-move))
            (increase (total-cost) 100)
        )
    )

    (:action move
        :parameters (?start - location ?end - location)
        :precondition (and
            (not (= ?start ?end))
            (not (restrict-move-to ?end))
            (not (ban-move))
            (rob-at ?start)
        )
        :effect (and
            (not (rob-at ?start))
            (rob-at ?end)
            (ban-move)
            (increase (total-cost) (known-cost ?start ?end))
        )
    )

    ;(:action find
    ;    :parameters (?obj - item)
    ;    :precondition (and
    ;        (not (is-located ?obj))
    ;        (is-pickable ?obj)
    ;        (hand-is-free)
    ;    )
    ;    :effect (and
    ;        (is-located ?obj)
    ;        (not (hand-is-free))
    ;        (is-holding ?obj)
    ;        (increase (total-cost) (find-cost ?obj))
    ;    )
    ;)

    (:action find
        :parameters (?obj - item ?from - location ?to - location)
        :precondition (and
            (not (restrict-move-to ?to))
            (rob-at ?from)
            (not (is-located ?obj))
            (is-pickable ?obj)
            (hand-is-free)
        )
        :effect (and
            (is-located ?obj)
            (not (hand-is-free))
            (is-holding ?obj)
            (rob-at ?to)
            (not (rob-at ?from))
            (not (ban-move))
            (increase (total-cost) (find-cost ?obj ?from ?to))
        )
    )

    (:action fill
        :parameters (?liquid - item ?loc - location ?cnt - item)
        :precondition (and
            (rob-at ?loc)
            (is-at ?liquid ?loc)
            (is-holding ?cnt)
            (not (is-dirty ?cnt))
            (is-liquid ?liquid)
            (is-fillable ?cnt)
            (forall (?i - item)
                (not (filled-with ?i ?cnt))
            )
        )
        :effect (and
            (filled-with ?liquid ?cnt)
        )
    )

    (:action pour
        :parameters (?liquid - item ?loc - location ?cnt - item)
        :precondition (and
            (rob-at ?loc)
            (is-liquid ?liquid)
            (is-fillable ?loc)
            (filled-with ?liquid ?cnt)
            (is-holding ?cnt)
        )
        :effect (and
            (is-at ?liquid ?loc)
            (not (filled-with ?liquid ?cnt))
        )
    )

    (:action make-coffee
        :parameters (?c - item)
        :precondition (and
            (rob-at coffeemachine)
            (is-at water coffeemachine)
            (is-at coffeegrinds coffeemachine)
            (is-fillable ?c)
            (not (is-dirty ?c))
            (is-at ?c coffeemachine)
            (not (filled-with water ?c))
            (not (filled-with coffee ?c))
        )
        :effect (and
            (filled-with coffee ?c)
            (not (is-at water coffeemachine))
            (not (is-at coffeegrinds coffeemachine))
            (is-dirty ?c)
        )
    )

    (:action turn-dishwasher-on
        :parameters ()
        :precondition (and
            (rob-at dishwasher)
            ; (exists (?i - item) (is-at ?i dishwasher))
        )

        :effect (and
            (forall
                (?i - item)
                (when (and (is-at ?i dishwasher) (is-dirty ?i))
                    (not (is-dirty ?i))
                )
            )
            (increase (total-cost) 50)
        )
    )

    )
    """
    return DOMAIN_PDDL

def pour_water(container=None):
    if container:
        str = f'(filled-with water {container})'
    else:
        str = '''
        (exists
            (?c - item)
                (and (filled-with water ?c))
        )
        '''
    return str


def serve_water(location, container=None):
    if container:
        str = f'(and (filled-with water {container}) (is-at {container} {location}))'
    else:
        str = f'''
        (exists
            (?c - item)
                (and (filled-with water ?c) (is-at ?c {location}))
        )
        '''
    return str


def make_coffee(container=None):
    if container:
        str = f'(filled-with coffee {container})'
    else:
        str = '''
        (exists
            (?c - item)
                (and (filled-with coffee ?c))
        )
        '''
    return str


def serve_coffee(location, container=None):
    if container:
        str = f'(and (filled-with coffee {container}) (is-at {container} {location}))'
    else:
        str = f'''
        (exists
            (?c - item)
                (and (filled-with coffee ?c) (is-at ?c {location}))
        )
        '''
    return str


def clean_something(object):
    str = f'(not (is-dirty {object}))'
    return str


def make_sandwich(spread=None):
    if spread:
        str = f'(spread-applied bread {spread})'
    else:
        str = '''
        (exists
            (?sprd - spread)
                (spread-applied bread ?sprd)
        )
        '''
    return str


def serve_sandwich(location, spread=None):
    if spread:
        str = f'(and (spread-applied bread {spread}) (is-at bread {location}))'
    else:
        str = f'''
        (exists
            (?sprd - spread)
                (and (spread-applied bread ?sprd) (is-at bread {location}))
        )
        '''
    return str


def set_napkin(location, napkin=None):
    if napkin:
        str = f'(and (is-folded {napkin}) (is-at {napkin} {location}))'
    else:
        str = f'''
        (exists (?npkn - napkin)
            (and (is-folded ?npkn) (is-at ?npkn {location}))
        )
        '''
    return str


def place_two_objects(locs, objs):
    t_str = f'(and (is-at {objs[0]} {locs[0]}) (is-at {objs[1]} {locs[1]}))'
    return t_str


def multiple_goal(goals):
    t_str = '(or'
    for goal in goals:
        t_str += f' {goal}'
    t_str += ')'
    return t_str

from planners.stripstream.stripstream import solve_pddlstream


def post_process(plan):
    for name, args in plan:
        print name, args


def main():
    plan = solve_pddlstream()
    if plan is not None:
        post_process(plan)


if __name__ == '__main__':
    main()
(define (stream namo)

  (:stream gen-grasp
    :inputs (?o)
    :domain (and (Pickable ?o))
    :outputs (?g ?q_pick)
    :certified (and (Grasp ?g)
                    (BaseConf ?q_pick)
                    (GraspTransform ?o ?g ?q_pick) ))

  (:stream gen-placement
    :inputs (?o ?g ?pick_q ?region)
    :domain (and (Pickable ?o)
                 (Grasp ?g)
                 (BaseConf ?pick_q)
                 (GraspTransform ?o ?g ?pick_q)
                 (Region ?region))
    :outputs (?place_q ?obj_pose)
    :certified (and (BaseConf ?place_q)
                    (Pose ?o ?obj_pose)
                    (PlaceConf ?o ?obj_pose ?place_q))
                    )

  (:stream gen-base-traj
    :inputs (?q_init ?q_goal)
    :domain (and
                 (BaseConf ?q_init)
                 (BaseConf ?q_goal))
    :outputs (?traj)
    :certified (and (BTraj ?q_init ?q_goal ?traj)))

  (:predicate (TrajPoseCollision ?obstacle ?obstacle_pose ?q_init ?q_goal ?traj)
   (and (BTraj ?q_init ?q_goal ?traj)
        (Pose ?obstacle ?obstacle_pose)
        )

    )
)

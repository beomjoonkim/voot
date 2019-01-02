(define (stream namo)

  (:stream gen-grasp
    :inputs (?o)
    :domain (and (Pickable ?o))
    :outputs (?grasp ?q_pick ?gconfig)
    :certified (and (Grasp ?grasp)
                    (BaseConf ?q_pick)
                    (GraspConfig ?gconfig)
                    (GraspTransform ?o ?grasp ?q_pick ?gconfig)
                    ))

  (:stream gen-placement
    :inputs (?o ?g ?pick_q ?g_config ?region)
    :domain (and (Pickable ?o)
                 (Grasp ?g)
                 (BaseConf ?pick_q)
                 (GraspTransform ?o ?g ?pick_q ?g_config)
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

  (:stream gen-base-traj-with-obj
    :inputs (?o ?g ?pick_q ?g_config ?q_init ?q_goal ?region)
    :domain (and (Pickable ?o)
                 (Grasp ?g)
                 (BaseConf ?q_init)
                 (BaseConf ?q_goal)
                 (GraspTransform ?o ?g ?pick_q ?g_config)
                 (Region ?region))
    :outputs (?traj)
    :certified (and
                    (BTrajWithObject ?o ?g ?pick_q ?q_init ?q_goal ?traj)
                    )
  )

  (:predicate (TrajPoseCollision ?obstacle ?obstacle_pose ?q_init ?q_goal ?traj)
   (and (BTraj ?q_init ?q_goal ?traj)
        (Pose ?obstacle ?obstacle_pose)
        )
  )

  (:predicate (TrajPoseCollisionWithObject ?holding_o ?grasp ?pick_q ?g_config ?placed_obj ?placed_o_pose ?q_init ?q_goal ?traj)
   (and
         (BTrajWithObject ?holding_o ?grasp ?pick_q ?q_init ?q_goal ?traj)
         (GraspTransform ?holding_o ?grasp ?pick_q ?g_config)
         (Pose ?placed_obj ?placed_o_pose)
         )
  )
)

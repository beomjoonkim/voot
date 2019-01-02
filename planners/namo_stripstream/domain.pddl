(define (domain namo)
    (:requirements :strips :equality)
    (:predicates (Pickable ?o)
                 (Grasp ?g)
                 (GraspConf ?gc)
                 (GraspTransform ?o ?g ?q ?g_config)
                 (EmptyArm)
                 (Holding ?o ?g ?gc)
                 (BaseConf ?q)
                 (PlaceConf ?o ?p ?q)
                 (AtPose ?o ?p)
                 (AtConf ?q)
                 (InRegion ?o ?r)
                 (Region ?r)
                 (Contained ?p ?r)
                 (Pose ?o ?p)
                 (Robot ?robot)
                 (Picked ?o ?g ?pick_q ?g_config)
                 (Placed ?o)
                 (ObjPoseInRegion ?o ?p ?region)
                 (BTraj ?q1 ?q2 ?traj)
                 (TrajPoseCollisionWithObject ?holding_o ?grasp ?pick_q ?placed_o ?placed_p ?place_q ?traj)
                 (UnsafeBTrajWithObject ?holding_o ?grasp ?pick_q ?place_q ?traj)
                 (UnsafeBTraj ?q_init ?q_goal ?traj)
                 (TrajPoseCollision ?obstacle ?obstacle_pose ?q_init ?q_goal ?traj)
                 (GraspConfig ?g_config)
                 (BTrajWithObject ?o ?g ?pick_q ?q_init ?q_goal ?traj)
                 )

    (:action pickup
    :parameters (?o ?g ?pick_q ?g_config)
    :precondition (and (EmptyArm)
                       (Pickable ?o)
                       (AtConf ?pick_q)
                       (GraspTransform ?o ?g ?pick_q ?g_config)
                       )
    :effect (and (Picked ?o ?g ?pick_q ?g_config) (not (EmptyArm))))

    (:action movebase
    :parameters (?q_init ?q_goal ?traj)
    :precondition (and (AtConf ?q_init)
                       (BaseConf ?q_goal)
                       (BTraj ?q_init ?q_goal ?traj)
                       (not (UnsafeBTraj ?q_init ?q_goal ?traj))
                       (EmptyArm)
                       )
    :effect (and (AtConf ?q_goal) (not (AtConf ?q_init))))

    (:action place
    :parameters(?obj ?grasp ?pick_q ?place_q ?place_obj_pose ?region ?g_config)
    :precondition (and
                       (AtConf ?place_q)
                       (Picked ?obj ?grasp ?pick_q ?g_config)
                       (PlaceConf ?o ?place_obj_pose ?place_q)
                       (ObjPoseInRegion ?o ?p ?region))
    :effect (and (EmptyArm) (not (Picked ?o ?g ?pick_q ?g_config)) (InRegion ?o ?region) (AtPose ?o ?placement) ))

    (:derived (UnsafeBTraj ?q_init ?q_goal ?traj) (
        exists (?obstacle ?obstacle_pose)
                 (and
                    (AtPose ?obstacle ?obstacle_pose)
                    (TrajPoseCollision ?obstacle ?obstacle_pose ?q_init ?q_goal ?traj))
    ))

)



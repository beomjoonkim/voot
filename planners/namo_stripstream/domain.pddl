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
                 (InRegion ?o )
                 (Region ?region)
                 (Contained ?p ?r)
                 (Pose ?o ?p)
                 (Robot ?robot)
                 (Picked ?o ?g ?pick_q ?g_config)
                 (Placed ?o)
                 (ObjPoseInRegion ?o ?obj_pose ?place_q ?region)
                 (BTraj ?q1 ?q2 ?traj)
                 (TrajPoseCollisionWithObject ?holding_o ?grasp ?pick_q ?g_config ?placed_obj ?placed_p ?q_init ?q_goal ?traj)
                 (UnsafeBTrajWithObject ?holding_o ?grasp ?pick_q ?g_config ?q_init ?q_goal ?traj)
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
                       (BaseConf ?q_init)
                       (BaseConf ?q_goal)
                       (BTraj ?q_init ?q_goal ?traj)
                       (not (UnsafeBTraj ?q_init ?q_goal ?traj))
                       (EmptyArm)
                       )
    :effect (and (AtConf ?q_goal) (not (AtConf ?q_init))))

    (:action movebase_with_object
    :parameters (?obj ?grasp ?pick_q ?g_config ?q_init ?q_goal ?traj)
    :precondition (and
                       (Picked ?obj ?grasp ?pick_q ?g_config)
                       (AtConf ?q_init)
                       (BaseConf ?q_goal)
                       (BTrajWithObject ?obj ?grasp ?pick_q ?q_init ?q_goal ?traj)
                       (not (UnsafeBTrajWithObject ?obj ?grasp ?pick_q ?g_config ?q_init ?q_goal ?traj) )
                       )
    :effect (and (AtConf ?q_goal) (not (AtConf ?q_init))))

    (:action place
    :parameters(?obj ?grasp ?pick_q ?g_config ?place_q ?place_obj_pose ?region )
    :precondition (and
                       (Pickable ?obj)
                       (GraspTransform ?obj ?grasp ?pick_q ?g_config)
                       (BaseConf ?place_q)
                       (Pose ?obj ?place_obj_pose)
                       (Region ?region)

                       ;(Picked ?obj ?grasp ?pick_q ?g_config)
                       (AtConf ?place_q)
                       (PlaceConf ?obj ?place_obj_pose ?place_q)
                       ;(ObjPoseInRegion ?obj ?obj_pose ?place_q ?region)
                       )
    :effect (and (EmptyArm) (not (Picked ?obj ?grasp ?pick_q ?g_config)) (InRegion ?obj) ))

    (:derived (UnsafeBTraj ?q_init ?q_goal ?traj) (
        exists (?obstacle ?obstacle_pose)
                 (and
                    (AtPose ?obstacle ?obstacle_pose)
                    (TrajPoseCollision ?obstacle ?obstacle_pose ?q_init ?q_goal ?traj))
    ))

    (:derived (UnsafeBTrajWithObject ?holding_o ?grasp ?pick_q ?g_config ?q_init ?q_goal ?traj) (
        exists (?placed_o ?placed_p)
                 (and
                        (AtPose ?placed_o ?placed_p)
                        (TrajPoseCollisionWithObject ?holding_o ?grasp ?pick_q ?g_config ?placed_o ?placed_p ?q_init ?q_goal ?traj))
    ))

)



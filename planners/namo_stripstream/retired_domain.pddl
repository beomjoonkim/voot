(define (domain convbelt)
    (:requirements :strips :equality)
    (:predicates (Pickable ?o)
                 (Grasp ?g)
                 (GraspConf ?gc)
                 (GraspTransform ?o ?g ?q)
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
                 (Picked ?o ?g ?pick_q)
                 (Placed ?o)
                 (ObjPoseInRegion ?o ?p ?region)
                 (BTrajWithObject ?o ?g ?pick_q ?q2 ?traj)
                 (TrajPoseCollisionWithObject ?holding_o ?grasp ?pick_q ?placed_o ?placed_p ?place_q ?traj)
                 (UnsafeBTrajWithObject ?holding_o ?grasp ?pick_q ?place_q ?traj)
                 )

    (:action pickup
    :parameters (?o ?g ?absq)
    :precondition (and (EmptyArm)
                       (Pickable ?o)
                       (Grasp ?g)
                       (BaseConf ?absq)
                       (GraspTransform ?o ?g ?absq)
                       )
    :effect (and (Picked ?o ?g ?absq) (not (EmptyArm))))

    (:action place
    :parameters(?obj ?grasp ?pick_q ?place_q ?place_obj_pose ?region)
    :precondition (and
                       (Picked ?obj ?grasp ?pick_q)
                       (GraspTransform ?obj ?grasp ?pick_q)
                       (PlaceConf ?o ?place_obj_pose ?place_q))
                       (ObjPoseInRegion ?o ?p ?region)
                       )
    :effect (and (EmptyArm) (not (Picked ?o ?g ?pick_q)) (InRegion ?o ?region) (AtPose ?o ?placement) (AtConf ?place_q))

    (:derived (UnsafeBTrajWithObject ?holding_o ?grasp ?pick_q ?place_q ?traj) (
        exists (?placed_o ?placed_p)
                 (and
                        (AtPose ?placed_o ?placed_p)
                        (TrajPoseCollision ?holding_o ?grasp ?pick_q ?placed_o ?placed_p ?place_q ?traj))
    ))

)



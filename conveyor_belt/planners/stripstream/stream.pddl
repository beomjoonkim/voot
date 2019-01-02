(define (stream convbelt)

  (:stream gen-grasp
    :inputs (?o)
    :domain (and (Pickable ?o))
    ;:outputs (?g ?gc ?relq ?absq)
    :outputs (?g ?absq)
    :certified (and (Grasp ?g)
                    (BaseConf ?absq)
                    (GraspTransform ?o ?g ?absq) ))

 (:stream gen-placement
    :inputs (?o ?g ?pick_q)
    :domain (and (Pickable ?o)
                 (Grasp ?g)
                 (BaseConf ?pick_q)
                 (GraspTransform ?o ?g ?pick_q))
    :outputs (?q2 ?p ?traj)
    :certified (and (BaseConf ?q2)
                    (Pose ?o ?p)
                    (PlaceConf ?o ?p ?q2)
                    (BTraj ?o ?g ?pick_q ?q2 ?traj)))

  (:predicate (TrajPoseCollision ?holding_o ?grasp ?pick_q ?placed_o ?placed_p ?place_q ?traj)
    (and
         (BTraj ?holding_o ?grasp ?pick_q ?place_q ?traj)
         (Pose ?placed_o ?placed_p)
         )
  )
)

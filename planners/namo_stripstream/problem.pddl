(define (problem pb2)
    (:domain blocksworld)
    (:objects a b)
    (:init
     (arm-empty))
    (:goal (not arm-empty)))
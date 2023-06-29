from enum import Enum

class ControlMode(str, Enum):
    Position="pos"
    PositionDelta="pos_delta"

class Observation(str, Enum):
    Pos="q"
    Vel="qdot"
    Acc="qacc"
    Force="f"

    PosDelta="dq"
    ForceDelta="df"

class ObsConfig:
    Q_DQ   = [
        Observation.Pos,
        Observation.PosDelta
    ]

    DF     = [
        Observation.ForceDelta, 
    ]

    F_DF   = [
        Observation.Force, 
        Observation.ForceDelta
    ]

    Q_F_DF = [
        Observation.Pos, 
        Observation.Force, 
        Observation.ForceDelta
    ]
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

    InCon="inC"
    HadCon="hadC"

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

    Q_F_DF_IN = [
        Observation.Pos, 
        Observation.Force, 
        Observation.ForceDelta,
        Observation.InCon
    ]

    Q_F_DF_HAD = [
        Observation.Pos, 
        Observation.Force, 
        Observation.ForceDelta,
        Observation.HadCon
    ]

    Q_F_DF_IN_HAD = [
        Observation.Pos, 
        Observation.Force, 
        Observation.ForceDelta,
        Observation.InCon,
        Observation.HadCon
    ]

    Q_VEL_F_DF_IN_HAD = [
        Observation.Pos, 
        Observation.Vel, 
        Observation.Force, 
        Observation.ForceDelta,
        Observation.InCon,
        Observation.HadCon
    ]
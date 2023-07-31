from enum import Enum

class ControlTask(str, Enum):
    Force="force"
    Position="position"

class ControlMode(str, Enum):
    Position="pos"
    PositionDelta="pos_delta"

class Observation(str, Enum):
    Pos="q"
    Vel="qdot"
    Acc="qacc"
    Force="f"
    Action="act"

    PosDelta="dq"
    ForceDelta="df"

    InCon="inC"
    HadCon="hadC"

    ObjVel="obj_vel"

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

    Q_VEL_F_DF_IN_HAD_ACT = [
        Observation.Pos, 
        Observation.Vel, 
        Observation.Force, 
        Observation.ForceDelta,
        Observation.InCon,
        Observation.HadCon,
        Observation.Action
    ]
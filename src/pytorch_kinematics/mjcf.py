from typing import Union, Optional, Dict

import numpy as np
import mujoco as mj
from mujoco._structs import _MjModelBodyViews as MjModelBodyViews

import pytorch_kinematics.transforms as tf
from . import chain
from . import frame

# Converts from MuJoCo joint types to pytorch_kinematics joint types
JOINT_TYPE_MAP = {
    mj.mjtJoint.mjJNT_HINGE: 'revolute',
    mj.mjtJoint.mjJNT_SLIDE: 'prismatic',
    mj.mjtJoint.mjJNT_FREE: 'free'
}

GEOM_TYPE_MAP = {
    mj.mjtGeom.mjGEOM_MESH: 'mesh',
    mj.mjtGeom.mjGEOM_BOX: 'box',
    mj.mjtGeom.mjGEOM_SPHERE: 'sphere',
    mj.mjtGeom.mjGEOM_CYLINDER: 'cylinder',
}


def mj_body_to_geoms(body: mj.MjsBody, as_visual: bool, device: str = 'cuda') -> list[frame.Visual]:
    # Find all geoms having body as parent
    visuals = []
    for geom in body.geoms:
        valid = (geom.contype == 0 and geom.conaffinity == 0) if as_visual \
            else (geom.contype > 0 or geom.conaffinity > 0)
        if valid:
            visuals.append(frame.Visual(name=geom.name,
                                        offset=tf.Transform3d(rot=geom.quat, pos=geom.pos).to(device),
                                        geom_type=GEOM_TYPE_MAP[geom.type],
                                        geom_param=geom.size.astype(np.float32)))
    return visuals


def _build_chain_recurse(mj_spec: mj.MjSpec, parent_frame: frame.Frame, parent_body: mj.MjsBody, device: str = 'cuda'):
    parent_frame.link.visuals = mj_body_to_geoms(parent_body, as_visual=True, device=device)
    parent_frame.link.collisions = mj_body_to_geoms(parent_body, as_visual=False, device=device)
    # iterate through all bodies that are children of parent_body
    for body in parent_body.bodies:
        n_joints = len(body.joints)
        if n_joints > 1:
            raise ValueError("composite joints are not supported")
        if n_joints == 1:
            joint = body.joints[0]
            child_joint = frame.Joint(joint.name, offset=tf.Transform3d(pos=joint.pos).to(device), axis=joint.axis,
                                      joint_type=JOINT_TYPE_MAP[joint.type],
                                      limits=(joint.range[0], joint.range[1]))
        else:
            child_joint = frame.Joint(body.name + "_fixed_joint")
        child_link = frame.Link(body.name, offset=tf.Transform3d(rot=body.quat, pos=body.pos).to(device),
                                visuals=mj_body_to_geoms(body, as_visual=True, device=device),
                                collisions=mj_body_to_geoms(body, as_visual=False, device=device))
        child_frame = frame.Frame(name=body.name, link=child_link, joint=child_joint)
        parent_frame.children.append(child_frame)
        _build_chain_recurse(mj_spec, child_frame, body)

    # iterate through all sites that are children of parent_body
    for site in parent_body.sites:
        site_link = frame.Link(site.name, offset=tf.Transform3d(rot=site.quat, pos=site.pos).to(device))
        site_frame = frame.Frame(name=site.name, link=site_link)
        parent_frame.children.append(site_frame)


def build_chain_from_mjcf(mjcf: str, body: Union[None, str, int] = None,
                          assets: Optional[Dict[str, bytes]] = None,
                          device: str = 'cuda') -> tuple[mj.MjSpec, chain.Chain]:
    """
    Build a Chain object from MJCF data.

    Parameters
    ----------
    mjcf : str
        MJCF string data.
    body : str or int, optional
        The name or index of the body to use as the root of the chain. If None, body idx=0 is used.
    device: str = 'cuda'

    Returns
    -------
    mj.MjModel
        MuJoCo model
    chain.Chain
        Chain object created from MJCF.
    """
    # NOTE: No need for spec compilation, which does mesh geoms centering/alignment,
    # which makes `mj.MjModel.geom('..').pos/quat` different from XML! Also, here we only need XML parsing!
    mj_spec = mj.MjSpec.from_file(mjcf, assets=assets) if mjcf.endswith('.xml') else (
        mj.MjSpec.from_string(mjcf, assets=assets))
    if body is None:
        root_body = mj_spec.bodies[0]
    else:
        root_body = mj_spec.body(body)
    root_frame = frame.Frame(root_body.name,
                             link=frame.Link(root_body.name,
                                             offset=tf.Transform3d(rot=root_body.quat, pos=root_body.pos)),
                             joint=frame.Joint()).to(device)
    _build_chain_recurse(mj_spec, root_frame, root_body)
    return mj_spec, chain.Chain(root_frame, device=device)


def build_serial_chain_from_mjcf(mjcf: str, end_link_name, root_link_name="",
                                 device: str = 'cuda') -> chain.SerialChain:
    """
    Build a SerialChain object from MJCF data.

    Parameters
    ----------
    mjcf : str
        MJCF string data.
    end_link_name : str
        The name of the link that is the end effector.
    root_link_name : str, optional
        The name of the root link.

    Returns
    -------
    chain.SerialChain
        SerialChain object created from MJCF.
    """
    mjcf_chain = build_chain_from_mjcf(mjcf, device=device)
    serial_chain = chain.SerialChain(mjcf_chain, end_link_name, root_link_name, device=device)
    return serial_chain

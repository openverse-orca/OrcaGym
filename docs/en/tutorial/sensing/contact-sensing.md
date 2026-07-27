# 🤝 Contact Sensing

Use contact force information as the robot's "tactile perception."

> For the contact query API, see [📡 State Query API](../robot_control/state-queries-api.md).

## Contact as Tactile Signals

Contact forces can provide rich information:

- **Grasp Detection** — whether contact is made
- **Force Control** — maintaining a target contact force
- **Surface Identification** — contact normal direction
- **Collision Detection** — unexpected collisions

## Contact Query Pipeline

```python
# 1. Get all active contacts
contacts = env.query_contact_simple()
# Returns: [{"geom1": 12, "geom2": 34, "dist": ..., "pos": ..., "frame": ...}, ...]

# 2. Filter contacts of interest (note: keys are lowercase "geom1"/"geom2")
target_body_id = env.model.body_name2id("robot_finger")
finger_contacts = [
    c for c in contacts
    if env.model.get_geom_body_id(c["geom1"]) == target_body_id
    or env.model.get_geom_body_id(c["geom2"]) == target_body_id
]

# 3. Get contact forces (by list index, not by an ID field in the contact dict)
if finger_contacts:
    contact_ids = list(range(len(contacts)))
    forces = env.query_contact_force(contact_ids)
    # Get the force for the corresponding contact
    for i, c in enumerate(finger_contacts):
        contact_idx = contacts.index(c)
        f = forces[contact_idx]
        normal_force = f[0]  # 0th component is the normal force
```

## Grasp Detection

```python
def is_grasped(env, finger_names: list[str], object_name: str) -> bool:
    """Check whether fingers are in contact with a target object"""
    contacts = env.query_contact_simple()
    object_id = env.model.body_name2id(object_name)
    finger_ids = [env.model.body_name2id(f) for f in finger_names]

    for c in contacts:
        geom1_body = env.model.get_geom_body_id(c["geom1"])
        geom2_body = env.model.get_geom_body_id(c["geom2"])
        bodies = {geom1_body, geom2_body}
        if object_id in bodies and any(f in bodies for f in finger_ids):
            return True
    return False
```

## Force Control

```python
def force_control(env, target_force: float = 10.0):
    """Simple force control: maintain a target contact force"""
    contacts = env.query_contact_simple()

    if not contacts:
        return np.zeros(env.model.nu)  # no contact

    contact_ids = list(range(len(contacts)))
    forces = env.query_contact_force(contact_ids)
    total_force = sum(np.linalg.norm(f[:3]) for f in forces.values())

    # PID force control
    force_error = target_force - total_force
    correction = force_error * 0.01

    ctrl = np.zeros(env.model.nu)
    # ... distribute correction to the appropriate actuators
    return ctrl
```

## Contact Information Summary

```python
def contact_summary(env) -> dict:
    """Generate a contact summary"""
    contacts = env.query_contact_simple()

    summary = {
        "total_contacts": len(contacts),
        "body_pairs": set(),
        "total_force": 0.0,
        "max_force": 0.0,
    }

    if contacts:
        contact_ids = list(range(len(contacts)))
        forces = env.query_contact_force(contact_ids)

        for i, c in enumerate(contacts):
            f = forces[i][:3]  # linear force component (contact frame)
            magnitude = np.linalg.norm(f)

            summary["total_force"] += magnitude
            summary["max_force"] = max(summary["max_force"], magnitude)
            summary["body_pairs"].add((
                env.model.get_geom_body_id(c["geom1"]),
                env.model.get_geom_body_id(c["geom2"]),
            ))

    return summary
```

## Standing Detection

```python
def is_standing(env, min_normal_force: float = 50.0) -> bool:
    """Check if the robot is standing (whether foot contact normal force is sufficient)"""
    contacts = env.query_contact_simple()
    if not contacts:
        return False
    contact_ids = list(range(len(contacts)))
    forces = env.query_contact_force(contact_ids)
    # The 0th component in contact frame is the normal force, significantly positive when standing
    max_normal = max(abs(f[0]) for f in forces.values())
    return max_normal > min_normal_force
```

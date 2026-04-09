import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orca_core import OrcaHand


def sign_symbol(value: float, eps: float = 1e-4) -> str:
    if value > eps:
        return "+"
    if value < -eps:
        return "-"
    return "0"


def expected_symbol(delta_joint: float, inverted: bool) -> str:
    # Non-inverted: motor delta follows joint delta. Inverted: opposite direction.
    if delta_joint == 0:
        return "0"
    if inverted:
        return "-" if delta_joint > 0 else "+"
    return "+" if delta_joint > 0 else "-"


def choose_target(current: float, rom_min: float, rom_max: float, delta: float) -> tuple[float, float] | None:
    # Prefer +delta if feasible, otherwise fall back to -delta to stay inside ROM.
    plus = current + delta
    if rom_min <= plus <= rom_max:
        return plus, delta

    minus = current - delta
    if rom_min <= minus <= rom_max:
        return minus, -delta

    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify joint sign mapping (joint_to_motor_map +/-) against observed motor direction."
    )
    parser.add_argument(
        "config_path",
        nargs="?",
        default="orca_core/models/v1/orcahand_left/config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--delta-deg",
        type=float,
        default=2.0,
        help="Joint perturbation magnitude in degrees (default: 2.0)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=8,
        help="Interpolation steps for each perturbation (default: 8)",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=0.01,
        help="Delay between interpolation steps in seconds (default: 0.01)",
    )
    parser.add_argument(
        "--settle-time",
        type=float,
        default=0.06,
        help="Wait time after each command before reading motors (default: 0.06)",
    )
    parser.add_argument(
        "--joint",
        type=str,
        default=None,
        help="Validate a single joint only (e.g., ring_abd)",
    )
    args = parser.parse_args()

    delta = float(args.delta_deg)
    if delta <= 0:
        raise ValueError("--delta-deg must be > 0")

    hand = OrcaHand(args.config_path)
    ok, msg = hand.connect()
    print(f"connect: {ok} {msg}")
    if not ok:
        return 1

    mismatch_count = 0
    tested = 0

    try:
        hand.init_joints(force_calibrate=False)
        hand.enable_torque()
        time.sleep(0.1)

        joints = hand.config.joint_ids
        if args.joint is not None:
            if args.joint not in joints:
                print(f"Unknown joint: {args.joint}")
                return 2
            joints = [args.joint]

        jpos0 = hand.get_joint_position().as_dict()
        mpos0 = hand.get_motor_pos(as_dict=True)

        print(
            "joint,motor_id,inverted_cfg,joint_delta_cmd,expected_motor_delta,observed_motor_delta,result"
        )

        for joint in joints:
            motor_id = hand.config.joint_to_motor_map[joint]
            inverted = hand.config.joint_inversion_dict.get(joint, False)

            if jpos0.get(joint) is None:
                print(f"{joint},{motor_id},{inverted},skip,skip,skip,SKIP(not calibrated)")
                continue

            rom_min, rom_max = hand.config.joint_roms_dict[joint]
            target_info = choose_target(jpos0[joint], rom_min, rom_max, delta)
            if target_info is None:
                print(f"{joint},{motor_id},{inverted},skip,skip,skip,SKIP(no safe delta in ROM)")
                continue

            target, delta_cmd = target_info
            exp = expected_symbol(delta_cmd, inverted)

            hand.set_joint_positions({joint: target}, num_steps=args.num_steps, step_size=args.step_size)
            time.sleep(args.settle_time)

            mpos1 = hand.get_motor_pos(as_dict=True)
            observed_delta = mpos1[motor_id] - mpos0[motor_id]
            obs = sign_symbol(observed_delta)
            result = "PASS" if obs == exp else "MISMATCH"

            print(
                f"{joint},{motor_id},{inverted},{delta_cmd:+.2f},{exp},{obs} ({observed_delta:+.4f}),{result}"
            )

            tested += 1
            if result == "MISMATCH":
                mismatch_count += 1

            # Return this joint to its start position before moving on.
            hand.set_joint_positions(
                {joint: jpos0[joint]}, num_steps=args.num_steps, step_size=args.step_size
            )
            time.sleep(args.settle_time)

            mpos0 = hand.get_motor_pos(as_dict=True)

    finally:
        try:
            hand.disable_torque()
        except Exception:
            pass
        hand.disconnect()

    print(f"summary: tested={tested}, mismatches={mismatch_count}")
    return 1 if mismatch_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())

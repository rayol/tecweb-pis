"""
Frame sanity check for an IMUNet-style dataset, to run before training on it.

`ProposedSequence` builds the global-frame features with

    init_rotor = ori_[0] * rv_[0].conj()      # constant, calibrated on ONE sample
    ori        = init_rotor * rv_             # device attitude in the ground-truth frame

and takes the targets from `pos_[:, :2]`. That only produces a learnable problem if

  1. `ori_*` and `pos_*` are expressed in the SAME coordinate frame, and
  2. that frame is Z-up, because the target treats x-y as the horizontal plane.

The IMUNet dataset violates (1) in its ARCore half: `read_data_s10.py` remaps the recorded
position to (x, -z, y) but leaves the pose quaternion in ARCore's frame, so `init_rotor`
becomes a wrong constant rotation and features and targets end up in two frames that differ
by an arbitrary per-sequence rotation -- including an arbitrary yaw, which the network cannot
learn around. This script looks for exactly that class of problem in a new dataset.

Checks, per sequence
--------------------
A. gravity tilt      angle between mean specific force in the feature frame and +Z.
                     Ground-truth free. On IMUNet: 0.09-1.79 deg for the 49 correct (Tango)
                     sequences, 34.86-115.63 deg for the 77 broken (ARCore) ones.
B. vertical axis      peak-to-peak range of each pos_ axis. The vertical axis of a walking
                     trajectory has by far the smallest range; if that is not pos_z, the
                     ground-truth frame is not Z-up and assumption (2) fails.
C. init_rotor drift  angle between `init_rotor * rv_` and `ori_` over the sequence, as median
                     and 90th percentile (the maximum saturates at 180 deg and is hostage to
                     a single bad row). Zero at t=0 by construction. This tests the assumption
                     init_rotor makes directly -- that `ori_` and `rv_` differ by a constant
                     rotation -- so it is the sharpest ground-truth-free check here: on IMUNet
                     the correct sequences sit at a median of ~6 deg (genuine relative drift
                     between the game rotation vector and the tracker) while the broken ones
                     are at ~120 deg.
D. residual rotation `estimate_alignment` from frame_alignment.py, split into its tilt and
                     yaw parts. NOTE the yaw half is only meaningful for gross errors:
                     measured on IMUNet sequences that are already correct it returns
                     3-18 deg of spurious yaw (tilt stays accurate to <2 deg). Anything under
                     --yaw_floor is reported but never treated as a fault.

Usage
-----
    ../.venv/Scripts/python.exe check_dataset_frames.py --root_dir <dataset>

Runs in the training virtualenv (.venv), not the export one.
"""
import argparse
import json
import os
import os.path as osp

import numpy as np
import pandas
import quaternion

from frame_alignment import estimate_alignment, gravity_tilt_deg, rotation_angle_deg

# Measured on the IMUNet dataset; see the module docstring.
# The yaw half of check D is noisy: on IMUNet sequences that are already correct it returns up
# to 18 deg of spurious yaw, so the floor sits above that. A threshold of 20 flags two correct
# Tango sequences at 20.8 deg, which is noise, not a fault.
DEFAULT_YAW_FLOOR_DEG = 25.0
DEFAULT_TILT_OK_DEG = 5.0
# Correct IMUNet sequences drift 6-10 deg between rv_ and ori_, which is normal; the threshold
# sits above that so only genuinely worse sequences are flagged.
DEFAULT_DRIFT_WARN_DEG = 15.0

_COLUMNS = {
    'gyro': ['gyro_x', 'gyro_y', 'gyro_z'],
    'acce': ['acce_x', 'acce_y', 'acce_z'],
    'pos': ['pos_x', 'pos_y', 'pos_z'],
    'ori': ['ori_w', 'ori_x', 'ori_y', 'ori_z'],
    'rv': ['rv_w', 'rv_x', 'rv_y', 'rv_z'],
}


def read_sequence(path):
    """Load one sequence directory, mirroring ProposedSequence.load()."""
    csv_path = osp.join(path, 'processed/data.csv')
    pkl_path = osp.join(path, 'processed/data.pkl')
    if osp.exists(csv_path):
        frame = pandas.read_csv(csv_path)
    elif osp.exists(pkl_path):
        frame = pandas.read_pickle(pkl_path)
    else:
        raise FileNotFoundError('no processed/data.csv or processed/data.pkl under {}'.format(path))

    missing = [c for group in _COLUMNS.values() for c in group if c not in frame.columns]
    if 'time' not in frame.columns:
        missing.append('time')
    if missing:
        raise KeyError('missing columns: {}'.format(', '.join(missing)))

    return {
        'ts': frame[['time']].values / 1e09,
        'gyro': frame[_COLUMNS['gyro']].values,
        'acce': frame[_COLUMNS['acce']].values,
        'pos': frame[_COLUMNS['pos']].values,
        'ori': quaternion.from_float_array(frame[_COLUMNS['ori']].values),
        'rv': quaternion.from_float_array(frame[_COLUMNS['rv']].values),
    }


def compute_features(seq):
    """Global-frame gyro+accel, exactly as ProposedSequence builds them."""
    init_rotor = seq['ori'][0] * seq['rv'][0].conj()
    ori = init_rotor * seq['rv']

    nz = np.zeros(seq['ts'].shape)
    gyro_q = quaternion.from_float_array(np.concatenate([nz, seq['gyro']], axis=1))
    acce_q = quaternion.from_float_array(np.concatenate([nz, seq['acce']], axis=1))
    gyro_glob = quaternion.as_float_array(ori * gyro_q * ori.conj())[:, 1:]
    acce_glob = quaternion.as_float_array(ori * acce_q * ori.conj())[:, 1:]
    return np.concatenate([gyro_glob, acce_glob], axis=1), init_rotor


def init_rotor_residual(seq, init_rotor):
    """Angle between `init_rotor * rv_` and `ori_`, in degrees, over the whole sequence."""
    delta = (init_rotor * seq['rv']) * seq['ori'].conj()
    w = np.abs(quaternion.as_float_array(delta)[:, 0])
    return np.degrees(2.0 * np.arccos(np.clip(w, 0.0, 1.0)))


def split_rotation(R):
    """Split a rotation into how far it tips +Z (tilt) and how far it turns about +Z (yaw)."""
    z = R @ np.array([0.0, 0.0, 1.0])
    tilt = float(np.degrees(np.arccos(np.clip(z[2], -1.0, 1.0))))
    yaw = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    return tilt, yaw


def diagnose(name, path, args):
    seq = read_sequence(path)
    features, init_rotor = compute_features(seq)
    ts = seq['ts'].reshape(-1)

    gravity = np.mean(features[:, 3:6], axis=0)
    residual = init_rotor_residual(seq, init_rotor)
    ranges = np.ptp(seq['pos'], axis=0)
    R = estimate_alignment(ts, seq['pos'], features)
    tilt_R, yaw_R = split_rotation(R)

    dt = np.diff(ts)
    result = {
        'name': name,
        'samples': int(len(ts)),
        'duration_s': float(ts[-1] - ts[0]),
        'rate_hz': float(1.0 / np.median(dt)) if len(dt) else float('nan'),
        'nonfinite': int(np.count_nonzero(~np.isfinite(features))),
        'nonmonotonic_ts': int(np.count_nonzero(dt <= 0)),
        # A
        'tilt_deg': gravity_tilt_deg(features),
        'gravity_dir': (gravity / np.linalg.norm(gravity)).tolist(),
        # B
        'pos_range_m': ranges.tolist(),
        'pos_vertical_axis': 'xyz'[int(np.argmin(ranges))],
        # C
        'drift_p50_deg': float(np.percentile(residual, 50)),
        'drift_p90_deg': float(np.percentile(residual, 90)),
        # D
        'residual_total_deg': rotation_angle_deg(R),
        'residual_tilt_deg': tilt_R,
        'residual_yaw_deg': yaw_R,
    }
    result['verdict'] = verdict(result, args)
    return result


def verdict(d, args):
    if d['nonfinite'] or d['nonmonotonic_ts']:
        return 'DADOS'
    if d['tilt_deg'] > args.tilt_ok and d['pos_vertical_axis'] != 'z':
        return 'NAO-Z-UP'
    if d['tilt_deg'] > args.tilt_ok:
        return 'FRAME-INCONSISTENTE'
    if d['drift_p50_deg'] > args.drift_warn:
        return 'ORI-RV-INCOMPATIVEL'
    if abs(d['residual_yaw_deg']) > args.yaw_floor:
        return 'YAW-SUSPEITO'
    if d['drift_p90_deg'] > args.drift_warn:
        return 'DERIVA'
    return 'OK'


_EXPLANATION = {
    'OK': 'features e targets no mesmo frame Z-up; init_rotor funciona',
    'FRAME-INCONSISTENTE': 'gravidade fora de +Z mas pos_z e vertical: ori_ e pos_ em frames diferentes',
    'NAO-Z-UP': 'o frame do ground truth nao e Z-up; o target pos_[:, :2] esta errado',
    'ORI-RV-INCOMPATIVEL': 'ori_ e rv_ nao diferem por uma rotacao constante; init_rotor nao tem sentido',
    'YAW-SUSPEITO': 'erro de yaw acima do piso de ruido do estimador',
    'DERIVA': 'game rotation vector e tracker derivam entre si ao longo da sequencia',
    'DADOS': 'NaN/inf nas features ou timestamps nao monotonicos',
}


def discover(root_dir, list_path):
    if list_path:
        with open(list_path) as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    names = [n for n in sorted(os.listdir(root_dir))
             if osp.isdir(osp.join(root_dir, n, 'processed'))]
    if not names:
        raise SystemExit('No sequence directories with a processed/ subfolder under {}'.format(root_dir))
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    repo_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
    parser.add_argument('--root_dir', type=str, default=osp.join(repo_root, 'Datasets/IMUNet_dataset'))
    parser.add_argument('--list', type=str, default=None,
                        help='sequence list file; default is every subdirectory containing processed/')
    parser.add_argument('--limit', type=int, default=0, help='only check the first N sequences')
    parser.add_argument('--tilt_ok', type=float, default=DEFAULT_TILT_OK_DEG,
                        help='gravity this far from +Z or less is considered aligned')
    parser.add_argument('--yaw_floor', type=float, default=DEFAULT_YAW_FLOOR_DEG,
                        help='estimated yaw below this is treated as estimator noise, not a fault')
    parser.add_argument('--drift_warn', type=float, default=DEFAULT_DRIFT_WARN_DEG,
                        help='init_rotor residual above this is reported as drift')
    parser.add_argument('--json', type=str, default=None, help='also write the full report here')
    args = parser.parse_args()

    names = discover(args.root_dir, args.list)
    if args.limit:
        names = names[:args.limit]

    print('Checking {} sequences under {}'.format(len(names), args.root_dir))
    print('(todos os angulos em graus)\n')
    header = '{:<32} {:>6} {:>7} {:>5} {:>7} {:>7} {:>8} {:>8}  {}'.format(
        'sequencia', 'tilt', 'vert', 'rate', 'drift50', 'drift90', 'res.tilt', 'res.yaw', 'veredito')
    print(header)
    print('-' * len(header))

    results, failures = [], []
    for name in names:
        path = osp.join(args.root_dir, name)
        try:
            d = diagnose(name, path, args)
        except Exception as exc:
            failures.append((name, str(exc)))
            print('{:<32} {}'.format(name[:32], 'ERRO: {}'.format(exc)))
            continue
        results.append(d)
        print('{:<32} {:>6.1f} {:>7} {:>5.0f} {:>7.1f} {:>7.1f} {:>8.1f} {:>8.1f}  {}'.format(
            name[:32], d['tilt_deg'], 'pos_' + d['pos_vertical_axis'], d['rate_hz'],
            d['drift_p50_deg'], d['drift_p90_deg'],
            d['residual_tilt_deg'], d['residual_yaw_deg'], d['verdict']))

    print()
    counts = {}
    for d in results:
        counts[d['verdict']] = counts.get(d['verdict'], 0) + 1
    print('Resumo ({} sequencias analisadas, {} com erro de leitura):'.format(len(results), len(failures)))
    for key in sorted(counts, key=lambda k: -counts[k]):
        print('  {:<22} {:>4}   {}'.format(key, counts[key], _EXPLANATION[key]))

    if results:
        print()
        print(recommendation(counts, results, args))

    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'args': vars(args), 'results': results,
                       'failures': [{'name': n, 'error': e} for n, e in failures]}, f, indent=2)
        print('\nRelatorio completo em {}'.format(args.json))


def recommendation(counts, results, args):
    total = len(results)
    lines = ['Recomendacao:']

    if counts.get('DADOS'):
        lines.append('  * {}/{} sequencias com NaN ou timestamps nao monotonicos. Corrija primeiro:'.format(
            counts['DADOS'], total))
        lines.append('    os demais diagnosticos nao sao confiaveis nelas.')

    if counts.get('NAO-Z-UP'):
        lines.append('  * {}/{} sequencias nao estao num frame Z-up. Isso NAO e corrigivel pelo'.format(
            counts['NAO-Z-UP'], total))
        lines.append('    frame_alignment.py, que assume gravidade em +Z. Rotacione pos_ E ori_ juntos')
        lines.append('    para Z-up no seu pre-processamento antes de treinar.')

    if counts.get('FRAME-INCONSISTENTE'):
        tilts = [d['tilt_deg'] for d in results if d['verdict'] == 'FRAME-INCONSISTENTE']
        ok_tilts = [d['tilt_deg'] for d in results if d['verdict'] == 'OK']
        lines.append('  * {}/{} sequencias tem ori_ e pos_ em frames diferentes (tilt {:.1f}-{:.1f}).'.format(
            len(tilts), total, min(tilts), max(tilts)))
        lines.append('    Melhor: corrija na origem (aplique o mesmo remap de eixos aos dois, ou a nenhum).')
        if ok_tilts and max(ok_tilts) + 1 < min(tilts) - 1:
            lines.append('    Alternativa: treine com o frame_alignment.py ativo e --align_tilt_deg entre')
            lines.append('    {:.0f} e {:.0f}, que separa os dois grupos deste dataset.'.format(
                max(ok_tilts) + 1, min(tilts) - 1))
        else:
            lines.append('    O frame_alignment.py so ajuda se houver um gap claro de tilt entre as boas')
            lines.append('    e as ruins; aqui nao ha, entao o gate por tilt nao vai separa-las.')

    if counts.get('ORI-RV-INCOMPATIVEL'):
        lines.append('  * {}/{} sequencias em que ori_ e rv_ nao diferem por uma rotacao constante'.format(
            counts['ORI-RV-INCOMPATIVEL'], total))
        lines.append('    (drift mediano > {:.0f}). O init_rotor nao tem como funcionar: os dois nao'.format(
            args.drift_warn))
        lines.append('    descrevem a mesma atitude fisica. Verifique se ori_ e rv_ vem do mesmo aparelho')
        lines.append('    e da mesma gravacao, e se algum passou por remap de eixos.')

    if counts.get('YAW-SUSPEITO'):
        lines.append('  * {}/{} sequencias com yaw acima de {:.0f}, mas gravidade alinhada. O tilt nao'.format(
            counts['YAW-SUSPEITO'], total, args.yaw_floor))
        lines.append('    detecta isso, entao o gate do frame_alignment.py tambem nao vai corrigi-las.')
        lines.append('    Investigue a origem do yaw antes de treinar.')

    if counts.get('DERIVA'):
        lines.append('  * {}/{} sequencias com deriva p90 > {:.0f} entre rv_ e ori_. Considere estimar o'.format(
            counts['DERIVA'], total, args.drift_warn))
        lines.append('    init_rotor por minimos quadrados sobre a sequencia toda em vez de rv_[0].')

    if counts.get('OK') == total:
        lines.append('  * Todas as sequencias passaram. Pode treinar com --no_align_frame.')

    return '\n'.join(lines)


if __name__ == '__main__':
    main()

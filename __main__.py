import json
import platform
import subprocess
import sys
import tempfile
import venv
from pathlib import Path
from typing import Tuple, Dict

import click
import numpy as np

VALID_RESULTS = {
    'img_001': np.array([37, 10, 15, 30, 28, 17, 10, 31, 9, 28, 23, 10]),
    'img_002': np.array([37, 10, 15, 30, 28, 17, 10, 31, 9, 28, 23, 10]),
    'img_003': np.array([16, 29, 24, 31, 27, 41, 15]),
    'img_004': np.array([16, 29, 24, 31, 27, 41, 15]),
    'img_005': np.array([13, 14, 35, 13, 15, 26, 29, 4]),
    'img_006': np.array([13, 14, 35, 13, 15, 26, 29, 4]),
    'img_007': np.array([16, 12, 27, 28, 14, 31]),
    'img_008': np.array([16, 12, 27, 28, 14, 31]),
    'img_009': np.array([28, 16, 21, 34, 23, 16]),
    'img_010': np.array([28, 16, 21, 34, 23, 16]),
}


@click.command()
@click.argument('applications_directory', type=click.Path(exists=True, file_okay=False))
@click.argument('input_file', type=click.Path(exists=True, dir_okay=False))
@click.argument('images_directory', type=click.Path(exists=True, file_okay=False))
@click.argument('output_directory', type=click.Path(exists=True, file_okay=False))
@click.option('--no-run', is_flag=True)
@click.option('--no-compute', is_flag=True)
def main(applications_directory: str, input_file: str, images_directory: str, output_directory: str, no_run: bool,
         no_compute: bool):
    applications_directory = Path(applications_directory)
    input_file = Path(input_file)
    images_directory = Path(images_directory)
    output_directory = Path(output_directory)

    if not no_run:
        states = run_applications(applications_directory, input_file, images_directory, output_directory)
        with open(output_directory / 'states.json', 'w') as states_file:
            json.dump(states, states_file)

    if not no_compute:
        results = compute_results(output_directory)
        with open(output_directory / 'results.json', 'w') as results_file:
            json.dump(results, results_file)


def run_applications(applications_directory: Path, input_file: Path, images_directory: Path,
                     output_directory: Path) -> Dict[str, str]:
    states = {}
    for applications_directory_entry in applications_directory.iterdir():
        if applications_directory_entry.is_dir():
            student_name, status = process_application_directory(applications_directory_entry, input_file,
                                                                 images_directory, output_directory)
            states[student_name] = status

    return states


def compute_results(output_directory: Path) -> Dict[str, float]:
    global_results = {}
    for student_output_directory in output_directory.iterdir():
        if not student_output_directory.is_dir():
            continue

        results_file_path: Path = student_output_directory / 'results.json'
        try:
            with results_file_path.open() as results_file:
                results = json.load(results_file)

            images_scores_sum = 0.0
            for image_name, valid_result in VALID_RESULTS.items():
                image_result = results[image_name] if image_name in results else 0
                image_result = image_result if len(image_result) == len(valid_result) else 0
                images_scores_sum += np.sum(np.abs(image_result - valid_result)) / np.sum(valid_result)

            score = images_scores_sum / len(VALID_RESULTS)
            global_results[student_output_directory.name] = score
        except Exception as e:
            print(f'{student_output_directory.name} failed: {e}', file=sys.stderr)

    return global_results


def process_application_directory(path: Path, input_file: Path, images_directory: Path,
                                  output_dir: Path) -> Tuple[str, str]:
    print(f'Processing "{path.name}"...')
    requirements_file = path / 'requirements.txt'
    if requirements_file.exists():
        print('Installing external dependencies...')
        temp_venv_dir = Path(tempfile.gettempdir()) / 'SiSWVenv'
        venv_builder = venv.EnvBuilder(system_site_packages=True, clear=True, with_pip=True)
        venv_builder.create(str(temp_venv_dir))
        is_windows = any(platform.win32_ver())
        interpreter = str(temp_venv_dir / 'scripts' / 'python') if is_windows else str(temp_venv_dir / 'bin' / 'python')
        try:
            subprocess.check_call([interpreter, '-m', 'pip', '-qqq', 'install', '-r', str(requirements_file)])
        except subprocess.CalledProcessError:
            return path.name, 'PIPFAILED'
    else:
        interpreter = sys.executable

    for application_file in path.iterdir():
        if application_file.name.endswith('.py'):
            student_name = application_file.name[:-3]
            student_output_dir = output_dir / student_name
            student_output_dir.mkdir(exist_ok=True)
            results_file = student_output_dir / 'results.json'
            stdout_file = student_output_dir / 'stdout'
            stderr_file = student_output_dir / 'stderr'

            print(f'Running "{student_name}"...')
            try:
                with stdout_file.open('w') as stdout, stderr_file.open('w') as stderr:
                    subprocess.run([interpreter, str(application_file), str(images_directory), str(input_file),
                                    str(results_file)], cwd=str(path), stdout=stdout, stderr=stderr, timeout=100)
            except subprocess.TimeoutExpired:
                return student_name, 'TIMEOUT'
            except subprocess.SubprocessError:
                return student_name, 'ERROR'

            return student_name, 'OK'

    return path.name, 'NOSCRIPT'


if __name__ == '__main__':
    main()

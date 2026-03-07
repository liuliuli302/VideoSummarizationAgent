from pathlib import Path

from setuptools import find_packages, setup


def read_requirements() -> list[str]:
    requirements_path = Path(__file__).with_name("requirements.txt")
    if not requirements_path.exists():
        return []

    requirements: list[str] = []
    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        requirements.append(stripped)
    return requirements


setup(
    name="video_agent",
    version="0.2.0",
    description="Research-oriented video summarization system with multi-agent reasoning",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=read_requirements(),
    python_requires=">=3.10",
)

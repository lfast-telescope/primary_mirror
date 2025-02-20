from setuptools import setup

setup(
    name="primary_mirror",
    version="0.1.2",
    py_modules=[
        "interferometer_utils",
        "LFAST_TEC_output",
        "LFAST_wavefront_utils",
        "tec_helper",
    ],
    install_requires=[
        "pandas",
        "datetime",
        "requests",
        "numpy",
        "matplotlib",
        "scipy",
        "h5py",
        "opencv-python",
        "hcipy",
    ],  # Add dependencies here
)

import setuptools

setuptools.setup(
    name="abm",
    version="1.0.0",
    author="Mirko Reul",
    author_email="public@reul.ch",
    description="ABM Implementation for the paper 'How Loyalty Trials Shape Allegiance to Political Order'",
    long_description_content_type="text/markdown",
    packages=['abm/'],
    license="https://mit-license.org/",
    install_requires=[
        'openpyxl',
        'xlsxwriter',
        'matplotlib',
        'numpy',
        'pandas',
        'scipy',
        'seaborn',
        'statsmodels'
    ],
    python_requires='>=3.6',
)

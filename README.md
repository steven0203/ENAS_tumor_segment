# ENAS for 3D brain tumor segment

## Setup Environment


Use pip to install environment:

    $ pip install -r requirements.txt
## Preprocess data

Unzip Brats 2015 or Brats 2018 data set, structure of the data folder:

    data_folder
    └───HGG
    │   │    ...
    │   │    ...
    │   │    ...
    └───LGG
    │   │    ...
    │   │    ...
    │   │    ...

Preprocess brats 2015 or 2018 data:

    #preprocess brats 2015
    $ python preprocess2015.py --data_path <data_folder>
    #preprocess brats 2018
    $ python preprocess2018.py --data_path <data_folder>


## Search architecture

Search for the architecture for Brats 2015 or Brats 2018:

    #search for the architecture for Brats 2015:
    $ bash search2015.sh
    #search for the architecture for Brats 2018:
    $ bash search2018.sh

The shared model and controller in logs:

    ENAS_tumor_segment
    └───logs
    │   └───<time>
    │   │    │     params.json
    │   │    │     ......
    │   │    │     ......
    │   ...
    │   ...
    │   ...
## Derive architecture

After training the shared model and controller, derive the the result model:

    # For Brats 2015
    $ bash derive2015.sh logs/<time>
    # For Brats 2018
    $ bash derive2018.sh logs/<time>

And we can get the derived architecture from ./log/<time>/derive_dag.log, like:

    ....
    ....
    ....
    [0, 'identity'], [0, 'avg pool'] .....
    [[0, 'identity'], [0, 'avg pool'].....
    [[0, 'identity'], [0, '3x3x3 dilation 2']....
    best_dag :[[0, 'identity'], [0, '3x3x3']....
## Train the searched model

Train the searched model for Brats 2015 or Brats 2018:

    # For Brats 2015
    $ bash tain2015.sh logs/<time>
    # For Brats 2018
    $ bash train2018.sh logs/<time>
## Evaluate the search model

Train the searched model for Brats 2015 or Brats 2018:

    # For Brats 2015
    $ bash evaluate2015.sh logs/<time>
    # For Brats 2018
    $ bash evaluate2018.sh logs/<time>


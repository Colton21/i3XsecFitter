import os, sys
import pandas as pd
from remove_overlap import check_overlap
import click

@click.command()
@click.option('--auto', is_flag=True)
def main(auto):

    if auto == False:
        raise NotImplementedError('Only auto mode is ready right now!')

    


if __name__ == "__main__":
    main()
##end

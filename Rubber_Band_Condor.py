import shelve
import numpy as np
import EMSC

filename = 'input.dat'

with shelve.open(filename) as shelf:

	input_ = shelf["data"]

del my_shelf



filename = 'output.dat'

rbc = TAT.Rubber_Band()

with shelve.open(filename) as shelf:

	shelf["corrected"] = input_.apply(lambda row: rbc.fit_transform(input_.columns, row) 
                                      , raw = True, axis = 1)
List input options for each unit test in _unit_tests.txt_ file.
Each must be sepparated by:
  * _#TEST# text_
where 'text' can be any text (some label/identifier)

then, run the python script (from same location as hartreeFock executable!)
e.g.,
  * _./src/tests/unit_tests.py_

results will be places in :
* _./src/tests/results.txt_

and previous results will be moved to :
* _./src/tests/results.txt_old.txt_

Carefully check that we don't ruin anything here before committing to master!

import ruptures as rpt
import numpy as np

def find_changes(signal, method):
    if method == "Pelt":
        algo = rpt.Pelt().fit(signal)
        points = algo.predict(pen=10)

    if method == "Dynp":
        algo = rpt.Dynp().fit(signal)
        points = algo.predict(n_bkps=2)

    if method == "Binseg":
        algo = rpt.Binseg().fit(signal)
        points = algo.predict(n_bkps=2)

    if method == "Window":
        algo = rpt.Window().fit(signal)
        points = algo.predict(n_bkps=2)

    return points

def rpt_plot(signal, points):
    rpt.display(signal, points)
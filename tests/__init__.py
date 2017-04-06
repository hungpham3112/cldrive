# Copyright (C) 2017 Chris Cummins.
#
# This file is part of cldrive.
#
# Cldrive is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# Cldrive is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with cldrive.  If not, see <http://www.gnu.org/licenses/>.
#
import numpy as np

from numpy import testing as nptest

import cldrive

ENV = cldrive.make_env()


def lol2np(list_of_lists):
    return np.array([np.array(x) for x in list_of_lists])


def almost_equal(l1, l2):
    for x, y in zip(l1, l2):
        nptest.assert_almost_equal(lol2np(x), lol2np(y))

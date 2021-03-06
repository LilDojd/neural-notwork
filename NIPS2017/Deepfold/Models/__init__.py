# Copyright (c) 2017 Jes Frellsen and Wouter Boomsma. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from .CartesianHighresModel import CartesianHighres
from .CubedSphereBandedModel import CubedSphereBandedModel
from .CubedSphereDenseModel import CubedSphereDenseModel
from .CubedSphereModel import CubedSphereModel
from .CubedSphereBandedDisjointModel import CubedSphereBandedDisjointModel
from .SphericalModel import SphericalModel

models = {"CubedSphereModel": CubedSphereModel,
          "CubedSphereBandedModel": CubedSphereBandedModel,
          "CubedSphereBandedDisjointModel": CubedSphereBandedDisjointModel,
          "CubedSphereDenseModel": CubedSphereDenseModel,
          "SphericalModel": SphericalModel,
          "CartesianHighresModel": CartesianHighres}

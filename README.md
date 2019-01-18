# Army-FCN-Python
Fully Cascaded Neural Network (FCN) Python code developed at US Army / AMRDEC


# Read Me for Fully Connected Neural Network (FCN) Python Code

_Distribution Statement A: For Public Release per AMRDEC PAO (20190023)_

Andrew Wilson, PhD

US Army AMRDEC, AED

This code is a basic but functional implementation of the Fully Connected Neural Network (FCN) using Neuron-by-Neuron (NBN) training, as described by Wilamowski (See references).

The code loosely follows the scikit-learn (https://scikit-learn.org) BaseEstimator interface, with the ultimate intention of providing fully compliant scikit-learn compatible Classifier and Regressor classes.

## Files
`fcn/__init__.py` – An empty file to turn this directory into a packaged module

`fcn/fcn.py` – This contains a reference implementation of FCN and NBN training. 

`fcn/fcn2.py` – This contains a higher-performance derived implementation which uses blocks to push loops into optimized numpy functions.

`fcn/fcn_minibatch.py` – An experimental variation which trains the network on mini-batches, but continues to use the full 2nd order Levenberg-Marquedt NBN training scheme.

`fcn/neurons.py` – Defines neuron class and associated activation functions, with required slope functions.

`fcn/feedforward.py` – A reference implementation of a simplified code for running an FCN in deployment without all the training machinery.

## Tests
`tests/test_fcn_square.py` – A standard test used by Wilamoski to demonstrate the performance of the FCN. The problem is to learn the boundaries of a spiral and correctly predict whether a pixel is inside or outside the spiral.

`tests/test_fcn_parity.py` - A standard test used by Wilamoski to demonstrate the performance of the FCN. Given a vector of bits which are each positive or negative, correctly predict the overall parity of the vector.

`tests/2_bit_parity.txt`

`tests/3_bit_parity.txt`

`tests/4_bit_parity.txt`

`tests/5_bit_parity.txt`

`tests/6_bit_parity.txt`

`tests/7_bit_parity.txt` – Data files for the parity test problem.

## References
 * B. Wilamowski, "Neural Network Architectures," in Industrial Electronics Handbook, vol. 5 – Intelligent Systems, CRC Press, 2011, pp. 6-1 - 6-17.
 * B. Wilamowski, "Neural Networks Learning," in Industrial Electronics Handbook, vol. 5 – Intelligent Systems, CRC Press, 2011, pp. 11-1 - 11-18.
 * B. Wilamowski, "Parity-N Problems as a Vehicle to Compare Efficiency of Neural Network Architectures," in Industrial Electronics Handbook, vol. 5 – Intelligent Systems, CRC Press, 2011, pp. 10-1 - 10-8.
 * B. Wilamowski, "Understanding of Neural Networks," in Industrial Electronics Handbook, vol. 5 – Intelligent Systems, CRC Press, 2011, pp. 5-1 - 5-11.
 * B. Wilamowski and H. Yu, "Neural Network Learning without Backpropagation," IEEE Transactions on Neural Networks, vol. 21, no. 11, pp. 1793-1803, 2010.
 * B. Wilamowski, N. Cotton, O. Kaynak and G. Dundar, "Computing Gradient Vector and Jacobian Matrix in Arbitrarily Connected Neural Networks," IEEE Transactions on Industrial Electronics, vol. 55, no. 10, pp. 3784-3790, 2008.
 * B. Wilamowski, H. Yu and N. Cotton, "NBN Algorithm," in Industrial Electronics Handbook, vol. 5 – Intelligent Systems, CRC Press, 2011, pp. 13-1 - 13-24.
 * H. Yu and B. Wilamowski, "Levenberg-Marquardt Training," in Industrial Electronics Handbook, vol. 5 – Intelligent Systems, CRC Press, 2011, pp. 12-1 - 12-16.
 * H. Yu, T. Xie and B. Wilamowski, "Neuro-Fuzzy System," in Industrial Electronics Handbook, vol. 5 – Intelligent Systems, CRC Press, 2011, pp. 20-1 - 20-9.


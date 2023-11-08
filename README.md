# ANN_RadialBP2CardiacOutput
## Cardiac output estimated from an uncalibrated radial blood pressure waveform: validation in an in-silico-generated population

This repository hosts scripts for the development and testing of a simple artificial neural network (ANN) model to predict cardiac output from non-invasive calibrated and uncalibrated radial blood pressure data.

**Abstract**
Background: Cardiac output is essential for patient management in critically ill patients. The state-of-the-art for cardiac output monitoring bears limitations that pertain to the invasive nature of the method, high costs, and associated complications. Hence, the determination of cardiac output in a non-invasive, accurate, and reliable way remains an unmet need. The advent of wearable technologies has directed research towards the exploitation of wearable-sensed data to improve hemodynamical monitoring.
Methods: We developed an artificial neural networks (ANN)-enabled modelling approach to estimate cardiac output from radial blood pressure waveform. In silico data including a variety of arterial pulse waves and cardiovascular parameters from 3,818 virtual subjects were used for the analysis. Of particular interest was to investigate whether the uncalibrated, namely, normalized between 0 and 1, radial blood pressure waveform contains sufficient information to derive cardiac output accurately in an in silico population. Specifically, a training/testing pipeline was adopted for the development of two artificial neural networks models using as input: the calibrated radial blood pressure waveform (ANNcalradBP), or the uncalibrated radial blood pressure waveform (ANNuncalradBP).
Results: Artificial neural networks models provided precise cardiac output estimations across the extensive range of cardiovascular profiles, with accuracy being higher for the ANNcalradBP. Pearson’s correlation coefficient and limits of agreement were found to be equal to [0.98 and (−0.44, 0.53) L/min] and [0.95 and (−0.84, 0.73) L/min] for ANNcalradBP and ANNuncalradBP, respectively. The method’s sensitivity to major cardiovascular parameters, such as heart rate, aortic blood pressure, and total arterial compliance was evaluated.
Discussion: The study findings indicate that the uncalibrated radial blood pressure waveform provides sample information for accurately deriving cardiac output in an in silico population of virtual subjects. Validation of our results using in vivo human data will verify the clinical utility of the proposed model, while it will enable research applications for the integration of the model in wearable sensing systems, such as smartwatches or other consumer devices.

<img width="1087" alt="Screenshot at Oct 19 18-30-10" src="https://github.com/Vicbi/ANN_RadialBP2CardiacOutput/assets/10075123/ce5a2116-e0d1-45e1-9c7e-743c39b03e45">

**Original Publication**

For a comprehensive understanding of the methodology and background, refer to the original publication: Bikia V, Rovas G and Stergiopulos N (2023) Cardiac output estimated from an uncalibrated radial blood pressure waveform: validation in an in-silico-generated population. Front. Bioeng. Biotechnol. 11:1199726.

**Citation**

If you use this code in your research, please cite the original publication:

Bikia V, Rovas G and Stergiopulos N (2023) Cardiac output estimated from an uncalibrated radial blood pressure waveform: validation in an in-silico-generated population. Front. Bioeng. Biotechnol. 11:1199726. doi: 10.3389/fbioe.2023.1199726

**License**

This project is licensed under the Apache License 2.0 - see the LICENSE.md file for details.

This work was developed as part of a research project undertaken by the Laboratory of Hemodynamics and Cardiovascular Technology at EPFL (https://www.epfl.ch/labs/lhtc/).


Feel free to reach out at vickybikia@gmail.com if you have any questions or need further assistance!


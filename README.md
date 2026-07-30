### Channel-estimation (PiPNet)

In multiple-input multiple-output systems, the acquisition of high-fidelity channel state information is hindered by the complex interaction between time-frequency fading and limited pilot signal resources. Current deep learning methods often lack physical consistency and require large computational costs. This paper proposes PiPNet, a lightweight physics-aware Transformer architecture that integrates specialized a priori knowledge into the model, including Pilot-Aware Encoding which establishes a sparsity-preserving manifold projection mechanism for noise filtering; the Hyper-Informed Physical Attention block which utilizes a physical triad comprising spectral, phase, and differential domains, combined with an inverted attention mechanism to reduce computational complexity; and the Phasor Rectification Feed-forward block to refine signal edge details. Simulation results demonstrate that PiPNet consistently achieves superior performance compared with existing methods. Furthermore, PiPNet exhibits an improved balance between estimation accuracy and computational efficiency, indicating its potential for real-time wireless applications.

![Architecture](https://github.com/DatChanThanh/Channel-estimation/blob/c0fef51d02f1a66ff74cabb77f3776775373eae5/architecture.png)

The dataset can be download on [Google Drive](https://drive.google.com/drive/folders/1FFxDl2-0lAaFuUiXNhAShjhNcGoqwvLD?usp=sharing) (please report if not available).

 If there is any error or need to be discussed, please email to [Thanh-Dat Tran](https://github.com/DatChanThanh) via [trandatt21@gmail.com](mailto:trandatt21@gmail.com).

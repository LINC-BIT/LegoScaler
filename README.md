# LegoScaler: Differentiated Block-grained Scaling for Mixed Inference and Retraining Jobs at Edge

<p align="center">
  <img src="./readme_imgs/intro.png" alt="介绍图" width="600" height="400">
</p>

This repository contains the artifacts for the paper **"LegoScaler: Differentiated Block-grained Scaling
for Mixed Inference and Retraining Jobs at Edge"**.

## Outline (Evaluation process/workflow and Reusability)

<a href="#1-artifact-overview">1. Artifact Overview</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#11-introduction">1.1 Introduction</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#12-hardwaresoftware-requirements-and-dependencies">1.2 Hardware/software Requirements and Dependencies</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#121-hardware-requirements">1.2.1 Hardware Requirements</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#122-software-requirements">1.2.2 Software Requirements</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#123-get-source-code">1.2.3 Get Source Code</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#124-install-dependencies">1.2.4 Install Dependencies</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#125-about-dataset">1.2.5 About Dataset</a><br>
<a href="#2-evaluation-reproduction">2. Evaluation Reproduction</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#21-evaluation-of-accuracy-predictor-figure-7-in-section-v-b">2.1 Evaluation of Accuracy Predictor (Figure 7 in Section V-B)</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#22-evaluation-of-knowledge-transfer-figure-8-a-in-section-v-b">2.2 Evaluation of Knowledge Transfer (Figure 8-a in Section V-B)</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#23-evaluation-of-model-generator-figure-8-b-in-section-v-b">2.3 Evaluation of Model Generator (Figure 8-b in Section V-B)</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#24-execution-time-breakdown-figure-8-c-in-section-v-b">2.4 Execution Time breakdown (Figure 8-c in Section V-B)</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#25-evaluation-of-multi-job-scheduling-at-edge-figure-9-in-section-v-c">2.5 Evaluation of Multi-job Scheduling at Edge (Figure 9 in Section V-C)</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#26-comparison-of-memory-footprint-figure-10-in-section-v-d">2.6 Comparison of Memory Footprint (Figure 10 in Section V-D)</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#27-comparison-of-energy-consumption-table-3-in-section-v-d">2.7 Comparison of Energy Consumption (Table 3 in Section V-D)</a><br>
<a href="#3-reusability-integrating-legoscaler-with-models-and-edge-schedulers">3. Reusability: Integrating LegoScaler with Models and Edge Schedulers</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#31-integrating-different-models">3.1 Integrating Different Models</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#32-integrating-different-edge-schedulers">3.2 Integrating Different Edge Schedulers</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#321-integrating-inference-oriented-schedulers">3.2.1 Integrating Inference-oriented Schedulers</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#322-integrating-retraining-oriented-schedulers">3.2.2 Integrating Retraining-oriented Schedulers</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#323-integrating-other-edge-schedulers">3.2.3 Integrating Other Schedulers</a><br>


## 1. Artifact Overview

### 1.1 Introduction<img src="./readme_imgs/heading-divider.svg" alt="" width="100%" height="1">

- **Background**:
  - **Edge AI applications**: Artificial intelligence (AI) applications such as object/image 
  recognition and question answering have become ubiquitous in 
  edge computing system.
  - **Large discrepancy in network architectures**: The inference model is designed for the stationary input distributions, 
  while the retraining model’s network architecture is built upon the current input distribution.
  - **Conflicting optimization objectives in resource allocation**: The objective of an inference model is to 
  maximize its accuracy under the latency constraint over the application’s 
  entire running period. But for the retraining model, it is scaled to maximize the accuracy after a short retraining window (e.g. 30 minutes).

- **LegoScaler Modules**: 
  - **Differentiated model generator**: It retains the most important components of 
  each model according to the current input distribution.
  - **Neuron-grained knowledge transfer**: It propagates the learned knowledge(i.e. the weights' change of neurons) from the 
  retraining model to the inference model through neuron indexes.
  - **Accuracy predictor**: It quantitatively evaluates the model accuracy under different configurations.
  - **Block-grained scheduler**: It selects the optimal scaling solutions and resource allocations that 
  maximize overall accuracy under the optimization constraints.

  ![](./readme_imgs/legoscaler.png)

- **Evaluation**: 
  - **Basic setting**: Our experiments compare 11 latest schedulers across 3 multi-application scenarios.
  - **Major results**: LegoScaler improves the overall accuracy average by 29.04%, reduces memory footprint by 37.79%
and energy consumption by 40.2%.

### 1.2 Hardware/software Requirements and Dependencies<img src="./readme_imgs/heading-divider.svg" alt="" width="100%" height="1">

#### 1.2.1 Hardware Requirements<img src="./readme_imgs/heading-divider-h4.svg" alt="" width="100%" height="1">

- **Hardware requirements for full running of experiments in the paper**:

  <table align="center">
    <thead>
      <tr>
        <th>RAM</th>
        <th>CPU</th>
        <th>Disk</th>
        <th>GPU</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>128 GB</td>
        <td>One 64-core server CPU (e.g., Intel(R) Xeon(R) Gold 6430)</td>
        <td>At least<br>150 GB free</td>
        <td>One NVIDIA GPU with more than 60 GB VRAM (e.g., A100)</td>
      </tr>
    </tbody>
  </table>

#### 1.2.2 Software Requirements<img src="./readme_imgs/heading-divider-h4.svg" alt="" width="100%" height="1">

- **Recommended software for full running of experiments in the paper**:

  <table align="center">
    <thead>
      <tr>
        <th>Operating System</th>
        <th>CUDA</th>
        <th>Others</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Ubuntu LTS 22.04.4 LTS</td>
        <td>CUDA 13.0</td>
        <td>Kernel 6.8.0-124-generic</td>
      </tr>
    </tbody>
  </table>

#### 1.2.3 Get Source Code<img src="./readme_imgs/heading-divider-h4.svg" alt="" width="100%" height="1">

  You can obtain the source code for artifacts evaluation by the following command. **The code does not perform any malicious or destructive operations**.

  ```bash
  git clone https://github.com/LINC-BIT/LegoScaler.git
  ```

#### 1.2.4 Install Dependencies<img src="./readme_imgs/heading-divider-h4.svg" alt="" width="100%" height="1">

  You can isntall the required dependencies by the following command.

  ```bash
  pip install -r requirements.txt
  ```

#### 1.2.5 About Dataset<img src="./readme_imgs/heading-divider-h4.svg" alt="" width="100%" height="1">

- **Dataset for pre-training**:
  After inserting FBS into blocks, the model is pre-trained on the following datasets.

  <table align="center">
    <thead>
      <tr>
        <th>Application</th>
        <th>Dataset</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Image classification</td>
        <td>ImageNet</td>
      </tr>
    </tbody>
  <tbody>
      <tr>
        <td>Object Detection</td>
        <td>COCO2017</td>
      </tr>
    </tbody>
  <tbody>
      <tr>
        <td>Text classification</td>
        <td>AGNews</td>
      </tr>
    </tbody>
  <tbody>
      <tr>
        <td>Visual question answering</td>
        <td>VQAv2</td>
      </tr>
    </tbody>
  </table>

- **Data points generation**:
  We train a specific accuracy predictor for each model. To train the predictor, we generate thousands of data points using **[EdgeVisionBench](https://github.com/LINC-BIT/EdgeVisionBench)**, which automatically constructs evolving distribution at edge.

- **Dataset for online scheduling**:
  The following datasets are randomly selected for online scheduling experiments.
    <table align="center">
    <thead>
      <tr>
        <th>Application</th>
        <th>Dataset</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Image classification</td>
        <td>CIFAR-10, STL-10, Caltech-256, Imagenette, Fashion-MNIST</td>
      </tr>
    </tbody>
  <tbody>
      <tr>
        <td>Object Detection</td>
        <td>COCO2017, VOC2012</td>
      </tr>
    </tbody>
  <tbody>
      <tr>
        <td>Text classification</td>
        <td>SST-2, IMDB, AGNews</td>
      </tr>
    </tbody>
  <tbody>
      <tr>
        <td>Visual question answering</td>
        <td>VQAv2-C, VQAv2's last 2129 classes</td>
      </tr>
    </tbody>
  </table>

## 2. Evaluation Reproduction

### 2.1 Evaluation of Accuracy Predictor (Figure 7 in Section V-B)<img src="./readme_imgs/heading-divider.svg" alt="" width="100%" height="1">

1. Generate data points for training accuracy predictor:
```bash
cd EdgeScheduler

python schedulers/predictor/scaling_law/cnn/gen_scaling_law_data_points.py
```

2. Train and evaluate the accuracy predictor:
```bash
python schedulers/predictor/scaling_law/scaling_law_trial/two_branch.py
```

The resource requirements and outputs are listed below:

<table align="center">
    <thead>
      <tr>
        <th>Resource Requirements</th>
        <th>Example Running Outputs</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>2 hours<br>10GB memory<br>25GB disk space</td>
        <td>
          <img src="./readme_imgs/fig_7_1.png" width="250">
          <img src="./readme_imgs/fig_7_2.png" width="300">
        </td>
      </tr>
    </tbody>
</table>

### 2.2 Evaluation of Knowledge Transfer (Figure 8-a in Section V-B)<img src="./readme_imgs/heading-divider.svg" alt="" width="100%" height="1">

This experiment evaluates how the knowledge learned by a retrained model flows back to the inference model through knowledge base.

  - **No feedback (`no`)**: The retraining model is discarded after its retraining window, and nothing is written back to the knowledge base.
  - **Direct replacement (`direct`)**: The retraining model directly replaces the inference model.
  - **Layer-wise feedback (`layer`)**: For each layer, the neuron weight changes produced by retraining are averaged into a single value, and this value is added back to every neuron of the corresponding layer in the knowledge base.
  - **Neuron-index feedback (`neuron`)**: Only the neurons actually trained in the retraining window are softly written back to their exact positions in the knowledge base through neuron indexes.

Commands for the 4 knowledge transfer strategies:
```bash
cd EdgeScheduler

# no feedback
python examples/two_classification_apps/main.py --knowledge_transfer no

# direct replacement
python examples/two_classification_apps/main.py --knowledge_transfer direct

# layer-wise feedback
python examples/two_classification_apps/main.py --knowledge_transfer layer

# neuron indexes
python examples/two_classification_apps/main.py --knowledge_transfer neuron
```

The resource requirements and outputs are listed below:

<table align="center">
    <thead>
      <tr>
        <th>Resource Requirements</th>
        <th>Example Running Outputs</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>2 hours<br>10GB memory<br>25GB disk space</td>
        <td>
          <img src="./readme_imgs/fig_8_1.png" width="250">
        </td>
      </tr>
    </tbody>
</table>

### 2.3 Evaluation of Model Generator (Figure 8-b in Section V-B)<img src="./readme_imgs/heading-divider.svg" alt="" width="100%" height="1">

This experiment evaluates how a scaled sub-model (its retained blocks/neurons) is generated when the block-grained scaling is performed: at a given density, which neurons are kept. The FBS modules predict the importance of each channel from the input they receive.

  - **Unimportant neurons (`unimportant`)**: Keep the least important neurons, i.e. the inverse of the importance-based selection (baseline).
  - **Random selection (`random`)**: Keep randomly chosen neurons (baseline).
  - **Importance on the source data (`source`)**: Measure the neuron importance by forwarding samples of the source (initial) dataset through the FBS modules, then keep the most important neurons.
  - **Importance on the current data (`current`, default)**: Measure the neuron importance on the samples of the current input distribution, then keep the most important neurons.

Commands for the 4 model generation strategies:
```bash
cd EdgeScheduler

# blocks with the least important neurons
python examples/two_classification_apps/main.py --model_generate unimportant

# blocks with randomly selected neurons
python examples/two_classification_apps/main.py --model_generate random

# blocks with the most important neurons measured on the source data
python examples/two_classification_apps/main.py --model_generate source

# blocks with the most important neurons measured on the current data
python examples/two_classification_apps/main.py --model_generate current
```

The resource requirements and outputs are listed below:

<table align="center">
    <thead>
      <tr>
        <th>Resource Requirements</th>
        <th>Example Running Outputs</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>2 hours<br>10GB memory<br>25GB disk space</td>
        <td>
          <img src="./readme_imgs/fig_8_2.png" width="300">
        </td>
      </tr>
    </tbody>
</table>

### 2.4 Execution Time breakdown (Figure 8-c in Section V-B)<img src="./readme_imgs/heading-divider.svg" alt="" width="100%" height="1">

This experiment breaks the wall time of one scheduling round of the whole pipeline down into its four steps:

  1. **Accuracy prediction**: evaluating candidate model-size configurations with the accuracy predictor inside the scheduler;
  2. **Model generation**: building the scaled sub-model from the channel importance predicted by the FBS modules;
  3. **Retraining**: the retraining loop of the retraining job;
  4. **Knowledge transfer**: writing the retrained knowledge back to the inference model.

Commands for testing the execution time:
```bash
cd EdgeScheduler

python examples/two_classification_apps/execution_time.py
```

The resource requirements and outputs are listed below:

<table align="center">
    <thead>
      <tr>
        <th>Resource Requirements</th>
        <th>Example Running Outputs</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>2 hours<br>10GB memory<br>25GB disk space</td>
        <td>
          <img src="./readme_imgs/fig_8_3.png" width="300">
        </td>
      </tr>
    </tbody>
</table>

### 2.5 Evaluation of Multi-job Scheduling at Edge (Figure 9 in Section V-C)<img src="./readme_imgs/heading-divider.svg" alt="" width="100%" height="1">

This experiment evaluates LegoScaler and the other schedulers on **multi-job edge workloads**, where several applications run their inference and retraining jobs concurrently under an evolving input distribution. The scheduler under test is selected with the `--scheduler` argument (default `ours`, i.e. LegoScaler).

Run one scenario under one scheduler:
```bash
cd EdgeScheduler

# run one scenario with LegoScaler (adjust the scenario by editing apps/apps_events in main.py)
python examples/two_classification_apps/main.py

# run the same scenario under another scheduler, e.g.:
python examples/two_classification_apps/main.py --scheduler EdgeOL
```

The resource requirements and outputs are listed below:

<table align="center">
    <thead>
      <tr>
        <th>Resource Requirements</th>
        <th>Example Running Outputs</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>1 hour<br>20GB memory<br>50GB disk space</td>
        <td>
          <img src="./readme_imgs/fig_9.png" width="250">
        </td>
      </tr>
    </tbody>
</table>

### 2.6 Comparison of Memory Footprint (Figure 10 in Section V-D)<img src="./readme_imgs/heading-divider.svg" alt="" width="100%" height="1">

During the online scheduling experiments, the memory footprint of each scheduler is recorded. 
```bash
cd EdgeScheduler

# run the online scheduling
python examples/two_classification_apps/main.py

# draw the memory footprint comparison figure
python examples/two_classification_apps/draw_pics/memory_footprint.py
```

The resource requirements and outputs are listed below:

<table align="center">
    <thead>
      <tr>
        <th>Resource Requirements</th>
        <th>Example Running Outputs</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>1 hour<br>20GB memory<br>50GB disk space</td>
        <td>
          <img src="./readme_imgs/fig_10.png" width="250">
        </td>
      </tr>
    </tbody>
</table>

### 2.7 Comparison of Energy Consumption (Table 3 in Section V-D)<img src="./readme_imgs/heading-divider.svg" alt="" width="100%" height="1">

During the online scheduling experiments, the energy consumption of each scheduler is recorded. 
```bash
cd EdgeScheduler

# run the online scheduling
python examples/two_classification_apps/main.py

# analysis and statistics of recorded data
python examples/two_classification_apps/draw_pics/energy_consumption.py
```

<table align="center">
    <thead>
      <tr>
        <th>Resource Requirements</th>
        <th>Example Running Outputs</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>1 hour<br>20GB memory<br>50GB disk space</td>
        <td>
          <img src="./readme_imgs/table_3.png" width="250">
        </td>
      </tr>
    </tbody>
</table>

## 3. Reusability: Integrating LegoScaler with Models and Edge Schedulers

LegoScaler can integrate various **models** (e.g. CNN and Transformer) and
**edge schedulers** (e.g. inference-oriented and retraining-oriented schedulers).

### 3.1 Integrating Different Models<img src="./readme_imgs/heading-divider.svg" alt="" width="100%" height="1">

- **Offline integration: build and pre-train an FBS version of your model.** The FBS module keeps the weights of an original layer and adds a channel-importance predictor, which is what enables block-grained scaling at runtime.

  - **Step 1: Convert the pre-trained model into an FBS model with `FBSModelConverter`.** Pass the architecture name via `model_type` (the converter dispatches per architecture, e.g. `vit`).

    ```python
    from EdgeScheduler.examples.two_classification_apps.FBS_nets.nets.create_fbs_model import FBSModelConverter

    model = XXX.from_pretrained('/path/to/pretrained/weights')  # e.g. ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

    converter = FBSModelConverter(density=1.0, model_type='XXX')  # 'XXX' = your architecture branch
    fbs_model = converter.convert_model(model)
    ```

    To support a new architecture, add a branch in `FBSModelConverter.convert_model` that wraps the layers/blocks to be scaled with `FBS(original_layer=..., in_channels=..., out_channels=..., density=..., model_type='XXX')`.

  - **Step 2: Jointly fine-tune the FBS model with `FBSJointTrainer`.** Dataloader functions follow the signature `get_XXX_dataloader(split, batch_size, model_type=None) -> (loader, dataset)`.

    ```python
    from EdgeScheduler.examples.two_classification_apps.FBS_nets.utils import FBSJointTrainer
    from EdgeScheduler.examples.two_classification_apps.data import get_XXX_dataloader

    train_loader, _ = get_XXX_dataloader('train', batch_size=64, model_type='XXX')
    val_loader, _   = get_XXX_dataloader('val',   batch_size=64, model_type='XXX')

    trainer = FBSJointTrainer(model=fbs_model, train_loader=train_loader, val_loader=val_loader,
                              device='cuda', num_classes=..., model_type='XXX', num_epochs=...)
    trainer.train(save_dir='/path/to/fbs_checkpoints')
    ```

- **Online integration: wrap the FBS model in an application actor.** The simulator only interacts with `ApplicationActor` and its training/inference jobs.

  - **Step 1: Create a subclass of `ApplicationActor`.** The base class launches and stops the jobs, publishes the latest model (`get_model_ref`) and accepts updates from training workers (`update_model`).

    ```python
    from EdgeScheduler.zraysched import ApplicationActor

    class Application_XXX(ApplicationActor):
        def __init__(self, app_name, training_job_actor_class, inference_job_actor_class, device):
            super().__init__(app_name, training_job_actor_class, inference_job_actor_class, device)
            self.origin_fbs_model = None
    ```

  - **Step 2: Implement the three hooks the base class needs**: 
    - `init_model()` returns the FBS model (on CPU, it is published via `ray.put`)
    - `get_fbs_model()` returns the full FBS model that is used to build scaled sub-models at runtime (cache it in `self.origin_fbs_model` like the demos);
    - `get_dataloader_func()` returns a dataloader function selected by `self.distribution_index`, which lets you rotate among datasets to emulate an evolving input distribution

    ```python
    def init_model(self):
        return torch.load('/path/to/fbs_checkpoints/xxx.pth', map_location='cpu')['main']

    def get_fbs_model(self):
        if self.origin_fbs_model is None:
            self.origin_fbs_model = self.init_model()
        return self.origin_fbs_model

    def get_dataloader_func(self):
        dataloaders_func = [get_cifar10_dataloader, get_caltech256_dataloader, ...]  # from data.py
        return dataloaders_func[self.distribution_index % len(dataloaders_func)]
    ```

  - **Step 3: Register the application and its events in `main.py`.**

    ```python
    from EdgeScheduler.examples.two_classification_apps.app_impl import Application_XXX
    from EdgeScheduler.examples.two_classification_apps.job_impl import DemoTrainingJob, DemoInferenceJob

    apps = dict(XXX=ray.remote(Application_XXX).remote('XXX', DemoTrainingJob, DemoInferenceJob, device=device), ...)

    # Event types: INFERENCE_START, INFERENCE_FINISH, TRAINING_START, TRAINING_FINISH
    apps_events = [AppEvent(app_id='XXX', timestamp=0, event_type=AppEventType.INFERENCE_START),
                   AppEvent(app_id='XXX', timestamp=0, event_type=AppEventType.TRAINING_START), ...]
    ```

    If your model needs special batching, optimization or evaluation (e.g. detection losses, tokenizers), extend the `self.model_type == 'XXX'` branches in `DemoTrainingJob.run_for` / `DemoInferenceJob.run_for` (`job_impl.py`).

- After the integration, the model can be used for online scheduling.

### 3.2 Integrating Different Edge Schedulers<img src="./readme_imgs/heading-divider.svg" alt="" width="100%" height="1">

#### 3.2.1 Integrating Inference-oriented Schedulers<img src="./readme_imgs/heading-divider-h4.svg" alt="" width="100%" height="1">

  - **AdaInf**: Interleave incremental retraining with inference based on the severity of data drift. 
    To this scheduler, you can set the `--scheduler` argument to `AdaInf` in the command line.
    ```bash
    python examples/two_classification_apps/main.py --scheduler AdaInf
    ```

  - **Corun**: Execute mixed jobs concurrently via spatial multiplexing.
    To this scheduler, you can set the `--scheduler` argument to `Corun` in the command line.
    ```bash
    python examples/two_classification_apps/main.py --scheduler Corun
    ```
  
  - **EdgeNN**: Accelerate inference jobs through semantic-aware memory management.
    To this scheduler, you can set the `--scheduler` argument to `EdgeNN` in the command line.
    ```bash
    python examples/two_classification_apps/main.py --scheduler EdgeNN
    ```

  - **ACBatch**: Optimize batching strategies via dynamic programming.
    To this scheduler, you can set the `--scheduler` argument to `ACBatch` in the command line.
    ```bash
    python examples/two_classification_apps/main.py --scheduler ACBatch
    ```

  - **MMSL**: Decomposes inference jobs via model partitioning.
    To this scheduler, you can set the `--scheduler` argument to `MMSL` in the command line.
    ```bash
    python examples/two_classification_apps/main.py --scheduler MMSL
    ```

  - **TS-MITO**: Optimize model selection and job offloading based on reinforcement learning.
    To this scheduler, you can set the `--scheduler` argument to `TS-MITO` in the command line.
    ```bash
    python examples/two_classification_apps/main.py --scheduler TS-MITO
    ```

  - **PSA**: Optimize model branch selection and communication resource allocation.
    To this scheduler, you can set the `--scheduler` argument to `PSA` in the command line.
    ```bash
    python examples/two_classification_apps/main.py --scheduler PSA
    ```

#### 3.2.2 Integrating Retraining-oriented Schedulers<img src="./readme_imgs/heading-divider-h4.svg" alt="" width="100%" height="1">

  - **AdaEvo**: Schedule multiple retraining jobs based on urgency.
    To this scheduler, you can set the `--scheduler` argument to `AdaEvo` in the command line.
    ```bash
    python examples/two_classification_apps/main.py --scheduler AdaEvo
    ```
    
  - **EdgeOL**: Improve the computational efficiency of retraining jobs according to Centered Kernel Alignment (CKA) similarity.
    To this scheduler, you can set the `--scheduler` argument to `EdgeOL` in the command line.
    ```bash
    python examples/two_classification_apps/main.py --scheduler EdgeOL
    ```
    
  - **SRS**: Insert retraining jobs into the Directed Acyclic Graph (DAG) of job requests.
    To this scheduler, you can set the `--scheduler` argument to `SRS` in the command line.
    ```bash
    python examples/two_classification_apps/main.py --scheduler SRS
    ```
        
  - **EdgeTA**: Perform neuron-grained model scaling and scheduling for retraining jobs.
    To this scheduler, you can set the `--scheduler` argument to `EdgeTA` in the command line.
    ```bash
    python examples/two_classification_apps/main.py --scheduler EdgeTA
    ```

#### 3.2.3 Integrating Other Edge Schedulers<img src="./readme_imgs/heading-divider-h4.svg" alt="" width="100%" height="1">

You can integrate a new edge scheduler into LegoScaler by the following steps. A scheduler interacts with the system through one unified interface that has three parts: **when** it is triggered, **how** it makes decisions, and **how** its decisions take effect. 

- **Step 1: Implement your scheduler class.** Create a file such as `EdgeScheduler/schedulers/retraining/my_scheduler.py`, subclass `Scheduler` (or `PeriodicScheduler`) from `EdgeScheduler.zraysched`, and fill in the three parts of the interface:

    - **Declare when the scheduler is triggered** in `reacted_events_type()`. Return the events that wake it up (e.g. `AppEventType.INFERENCE_START`) for event-driven scheduling; return `SchedulingTiming.EACH_WINDOW` to decide every time window; or return `SchedulingTiming.PERIODIC` to decide at a fixed interval.
    - **Implement the decision logic** in `async run(self, jobs)`, where `jobs` is `{job_id: job}` of all currently running jobs. A `job_id` follows the form `{app_name}-training` / `{app_name}-inference`, so you can tell the job type with `'train' in job_id` and the model name with `job_id.split('-')[0]`.
    - **Express the decisions through the return value** of `run()`: a dict `{job_id: {...}}`. Each entry supports `max_gpu_utilization` (the fraction of the next time window the job is allowed to run) and an optional `hyps` dict that is passed to the job's `run_for` (e.g. `batch_size`/`lr` for training; `model_size` for the block-grained scaling of LegoScaler).

    ```python
    from EdgeScheduler.zraysched import Scheduler, AppEventType, SchedulingTiming

    class MyScheduler(Scheduler):
        def reacted_events_type(self):
            return [AppEventType.INFERENCE_START, AppEventType.TRAINING_START]

        async def run(self, jobs):
            res = {}
            for job_id, job in jobs.items():
                if 'train' not in job_id:
                    continue
                # your scheduling idea: decide how much GPU time and which hyper-parameters each training job should get
                res[job_id] = {'max_gpu_utilization': 0.5,
                               'hyps': {'batch_size': 64, 'lr': 3e-4}}
            return res
    ```

- **Step 2: Register the scheduler** by exporting the class in `EdgeScheduler/schedulers/retraining/__init__.py`:

    ```python
    from .my_scheduler import MyScheduler
    ```

- **Step 3: Add a selection branch in the example driver.** In `main.py`, import the class and add an entry to the scheduler-selection code. 

    ```python
    from EdgeScheduler.schedulers.retraining.my_scheduler import MyScheduler

    # in the scheduler-selection part of main()
    elif scheduler_name == "my_scheduler":
        scheduler = ray.remote(num_gpus=0.1)(MyScheduler).remote()
    ```

- After the integration, you can run the new scheduler:

    ```bash
    cd EdgeScheduler

    python schedulers/examples/two_classification_apps/main.py --scheduler my_scheduler
    ```
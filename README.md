# ComfyUI_CogView4_Wrapper


The unofficial implementation of [CogView4](https://github.com/THUDM/CogView4) project in ComfyUI.    
Recommended to run on Nvidia GPU with VRAM of 16GB or more.
![image](image/cogview4_wrapper_example.jpg)

### Install
Open the cmd window in the plugin directory of ComfyUI, like ```ComfyUI\custom_nodes```，type  
```
git clone https://github.com/chflame163/ComfyUI_CogView4_Wrapper.git
```
Install dependency packages：
```
pip install -r ComfyUI_CogView4_Wrapper/requirements.txt
```

The model will be automatically downloaded to the ```ComfyUI/models/CogView/CogView4-6B``` directory during the first run.    

You can also manually download the model from [huggingface.co/THUDM/CogView4-6B](https://huggingface.co/THUDM/CogView4-6B/tree/main) or [BaiduNetdisk](https://pan.baidu.com/s/1mGbEq689Ncpc0QEDngO5Tg?pwd=id3m) to the above directory.

### Note
ComfyUI_CogView4-Wrapper requires the installation of the diffusers dependency package in dev version, which may affect certain plugins. Please be aware of backing up the environment.
    

## statement
This nodes follows the MIT license, Some of its functional code comes from other open-source projects. Thanks to the original author. If used for commercial purposes, please refer to the original project license to authorization agreement.
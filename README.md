this project refer to APES: Attention-based Point Cloud Edge Sampling  
[github link](https://github.com/JunweiZheng93/APES)  
rewrite with pytorch lighting

### APES-lighting
- train.py
    - model_interface.py
        - forward()
            - backbone()
            - head()
            - render()
        - training_step()
        - validation_step()
        - test_step()
    - data_interface.py
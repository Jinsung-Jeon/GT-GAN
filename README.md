# GT-GAN: General Purpose Time Series Synthesis with Generative Adversarial Networks


# Usage
## Install the environment using yaml file
~~~
conda env create -f environment.yaml

pip install numba==0.54.0 patsy==0.5.2 sktime==0.4.0 statsmodels==0.13.0 tf-slim==1.1.0 torchdiffeq==0.2.2 tqdm==4.62.3
~~~

## Train model 

### regular Stocks dataset
~~~
python GTGAN_stocks.py --dims 32-64-64-32 --train --atol 1e-2 --rtol 1e-3 --activation exp --max-steps 8500 --last_activation_r softplus --last_activation_d identity --reconstruction 0.01 --kinetic-energy 0.01 --jacobian-norm2 0.05 --directional-penalty 0.01 
~~~

### irregular Stocks dataset (dropped 70%)
~~~
python GTGAN_stocks_irregular.py --train --data stock --dims 32-64-64-32 --atol 1e-2 --rtol 1e-3 --first_epoch 10000 --max-steps 2000 --activation exp --last_activation_r softplus --last_activation_d identity --reconstruction 0 --kinetic-energy 0.05 --jacobian-norm2 0.01 --directional-penalty 0.05 --missing_value 0.7
~~~

### regular Energy dataset
~~~
python GTGAN_energy.py --data energy --train --atol 1e-3 --rtol 1e-3 --max-steps 6000 --log_time 2 --missing_value 0.0 --reconstruction 0.01 --kinetic-energy 0.5 --jacobian-norm2 0.1 --first_epoch 5000 --save_dir regular_energy
~~~

### irregular Energy dataset (dropped 50%)
~~~
python GTGAN_energy.py --data energy --train --atol 1e-3 --rtol 1e-3 --log_time 2 --max-steps 8500 --missing_value 0.5 --reconstruction 0.01 --kinetic-energy 0.5 --jacobian-norm2 0.1 --first_epoch 5000 --save_dir irregular_energy
~~~

## Test model
### regular Stocks dataset
~~~
python GTGAN_stocks.py --dims 32-64-64-32 --atol 1e-2 --rtol 1e-3 --activation exp --last_activation_r softplus --last_activation_d identity --reconstruction 0.01 --kinetic-energy 0.01 --jacobian-norm2 0.05 --directional-penalty 0.01 
~~~

### irregular Stocks dataset (dropped 70%)
~~~
python GTGAN_stocks_irregular.py --data stock --dims 32-64-64-32 --atol 1e-2 --rtol 1e-3 --max-steps 10000 --activation exp --last_activation_r softplus --last_activation_d identity --reconstruction 0 --kinetic-energy 0.05 --jacobian-norm2 0.01 --directional-penalty 0.05 --missing_value 0.7
~~~
### regular Energy dataset
~~~
python GTGAN_energy.py --data energy --atol 1e-3 --rtol 1e-3 --log_time 2 --max-steps 6000 --missing_value 0.0 --reconstruction 0.01 --kinetic-energy 0.5 --jacobian-norm2 0.1 --save_dir regular_energy
~~~
### irregular Energy dataset (dropped 50%)
~~~
python GTGAN_energy.py --data energy --atol 1e-3 --rtol 1e-3 --log_time 2 --max-steps 8500 --missing_value 0.5 --reconstruction 0.01 --kinetic-energy 0.5 --jacobian-norm2 0.1 --save_dir irregular_energy
~~~

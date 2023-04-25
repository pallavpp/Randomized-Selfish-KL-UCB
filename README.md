# Randomized Selfish-KL-UCB

This code implements the Randomized Selfish KL-UCB algorithm for multi-armed bandits. The original paper can be found [here](https://arxiv.org/abs/2102.10200). All results get stored in the `results` folder.

## Results Obtained:

|   | (μ_min = 0.01, μ_max = 0.99) | (μ_min = 0.1, μ_max = 0.2) | (μ_min = 0.8, μ_max = 0.9) |
|:-:|              :-:             |             :-:            |             :-:            |
| (M = 2, K = 3) | ![M_2_K_3_mu_min_0 01_mu_max_0 99](https://user-images.githubusercontent.com/62967830/234250080-941b09ee-21cc-4473-a529-38a5a5f1672d.png) | ![M_2_K_3_mu_min_0 1_mu_max_0 2](https://user-images.githubusercontent.com/62967830/234250124-d33fb4d1-6069-4eb4-82f5-eb3c1899ad93.png) | ![M_2_K_3_mu_min_0 8_mu_max_0 9](https://user-images.githubusercontent.com/62967830/234250146-722a7d9a-7d94-4f48-87ea-2e96b99e6993.png) |
| (M = 2, K = 5) | ![M_2_K_5_mu_min_0 01_mu_max_0 99](https://user-images.githubusercontent.com/62967830/234250360-18fbd209-492e-4cf0-b3dd-c53e19e8526f.png) | ![M_2_K_5_mu_min_0 1_mu_max_0 2](https://user-images.githubusercontent.com/62967830/234250381-c6f803b2-624b-421b-8720-3fdefb1b71cc.png) | ![M_2_K_5_mu_min_0 8_mu_max_0 9](https://user-images.githubusercontent.com/62967830/234250394-1388bbda-3625-4455-88a0-d5fd61175049.png) |
| (M = 5, K = 10) | ![M_5_K_10_mu_min_0 01_mu_max_0 99](https://user-images.githubusercontent.com/62967830/234250506-cc442d8a-89c3-4fee-87b5-8f6182d8df1f.png) | ![M_5_K_10_mu_min_0 1_mu_max_0 2](https://user-images.githubusercontent.com/62967830/234250519-5fd541c2-aa84-4ebc-beaf-7dd9b444f301.png) | ![M_5_K_10_mu_min_0 8_mu_max_0 9](https://user-images.githubusercontent.com/62967830/234250536-23cf0c6f-4daf-48e8-9762-37a68e689370.png) |
| (M = 10, K = 15) | ![M_10_K_15_mu_min_0 01_mu_max_0 99](https://user-images.githubusercontent.com/62967830/234250781-84160a54-2844-448a-8a11-55862ca551f5.png) | ![M_10_K_15_mu_min_0 1_mu_max_0 2](https://user-images.githubusercontent.com/62967830/234250794-7f2ff0de-2e40-4030-aecc-f70182d38076.png) | ![M_10_K_15_mu_min_0 8_mu_max_0 9](https://user-images.githubusercontent.com/62967830/234250809-e9d69816-e0e4-431f-abb3-4beae55ff32e.png) |

<p align="center" width="100%">
    <img width="50%" src="https://user-images.githubusercontent.com/62967830/234247081-2a682d91-0df8-467d-b427-9dafe2854c9d.png">
</p>

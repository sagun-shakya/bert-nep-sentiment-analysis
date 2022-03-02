# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 17:20:45 2022

@author: Sagun Shakya
"""
import torch


if __name__ == "__main__":
    # GPU Support.
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    print("Device: ",device)
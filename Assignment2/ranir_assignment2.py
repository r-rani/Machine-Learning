{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "c33b5e19-49b1-4c89-ac19-26320c1d63ab",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.neighbors import KNeighborsRegressor\n",
    "from sklearn.metrics import mean_squared_error"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "0b2b5667-ff6a-4dcf-9a79-bcfa1fbd01f1",
   "metadata": {},
   "outputs": [],
   "source": [
    "#variables \n",
    "seed = 1161\n",
    "N_train = 201\n",
    "N_test = 101\n",
    "K = np.arange(1,61)\n",
    "N_fold = 5"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "70023c3e-a0a1-4e0b-8936-cc0b61b29a63",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Equations \n",
    "def f_opt(x):\n",
    "    return np.sin(2.0 * np.pi * x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "4f7cb015-ae9c-4161-bd31-31e21a203a36",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Generate the test, vaildation, and training set \n",
    "def generate_sets(seed, N_train, N_test):\n",
    "    x_tr = np.linspace(0, 1, N_train)\n",
    "    x_te =np.linspace(0, 1, N_test)\n",
    "    np.random.randn(seed)\n",
    "    t_tr = f_opt(x_tr) + 0.2*np.random.randn(N_train)\n",
    "    t_te = f_opt(x_te) + 0.2*np.random.randn(N_test)\n",
    "    return x_tr, t_tr, x_va, t_va, x_te, t_te"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "997cfe22-42ff-4d85-bd47-c6cbcb8af01d",
   "metadata": {},
   "outputs": [
    {
     "ename": "_IncompleteInputError",
     "evalue": "incomplete input (3672948396.py, line 2)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  Cell \u001b[1;32mIn[11], line 2\u001b[1;36m\u001b[0m\n\u001b[1;33m    \u001b[0m\n\u001b[1;37m    ^\u001b[0m\n\u001b[1;31m_IncompleteInputError\u001b[0m\u001b[1;31m:\u001b[0m incomplete input\n"
     ]
    }
   ],
   "source": [
    "def eucladian_distance (x, t, k):\n",
    "    #sqrt((x1-x2)^2 + (y1-y2)^2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "edb1ad4e-a4b6-4f65-8d9e-2ea33e049a41",
   "metadata": {},
   "outputs": [],
   "source": [
    "def find_midpoint(x, xp, k):\n",
    "    return (np.sum(x-xp))/k"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9cd8e118-b214-4e50-8ad5-75ba910cdede",
   "metadata": {},
   "outputs": [],
   "source": [
    "def k_function(k,x_q, x_tr):\n",
    "    "
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python (base)",
   "language": "python",
   "name": "myenv"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

import numpy as np
import pytest
from src.data import one_hot

#one hot encoding tests
def test_one_hot():
    '''general unit test for correct position encoding'''
    unique_aa = ['A', 'C', 'G', 'T']
    protein_sequences = np.array(['ACGT', 'CGTA', 'TGCA'])
    encoded_sequences = one_hot(unique_aa, protein_sequences)
    expected_result = np.array([[[1, 0, 0, 0],  
                                [0, 1, 0, 0],  
                                [0, 0, 1, 0],  
                                [0, 0, 0, 1]], 

                                [[0, 1, 0, 0],  
                                [0, 0, 1, 0],  
                                [0, 0, 0, 1],  
                                [1, 0, 0, 0]],

                                [[0, 0, 0, 1], 
                                [0, 0, 1, 0],  
                                [0, 1, 0, 0], 
                                [1, 0, 0, 0]]])
    np.testing.assert_array_equal(encoded_sequences, expected_result)


def test_one_hot_uneven():
    '''testing uneven sequences'''
    unique_aa = ['A', 'C', 'G', 'T']
    protein_sequences = np.array(['ACGT', 'CGT', 'TG'])
    try:
        encoded_sequences = one_hot(unique_aa, protein_sequences)
        assert False
    except AssertionError: #inhomogenous sequence error
        assert True
        
def test_one_hot_missing():
    '''testing uneven sequences'''
    unique_aa = ['A', 'C', 'G', 'T']
    protein_sequences = np.array(['AAAA', 'CGTX', 'TG--'])
    encoded_sequences = one_hot(unique_aa, protein_sequences)
    expected_result = np.array([[[1, 0, 0, 0],  
                                [1, 0, 0, 0],  
                                [1, 0, 0, 0],  
                                [1, 0, 0, 0]], 

                                [[0, 1, 0, 0],  
                                [0, 0, 1, 0],  
                                [0, 0, 0, 1],  
                                [0, 0, 0, 0]],

                                [[0, 0, 0, 1], 
                                [0, 0, 1, 0],  
                                [0, 0, 0, 0], 
                                [0, 0, 0, 0]]])
    np.testing.assert_array_equal(encoded_sequences, expected_result)

def test_one_hot_empty():
    '''testing in case of empty sequences'''
    unique_aa = ['A', 'C', 'G', 'T']
    protein_sequences = np.array([])
    try:
        encoded_sequences = one_hot(unique_aa, protein_sequences)
        assert False
    except ValueError:
        assert True

def test_one_hot_empty1():
    '''testing in case of empty mapping'''
    unique_aa = []
    protein_sequences = np.array(['ACGT', 'CGTA', 'TGCA'])
    encoded_sequences = one_hot(unique_aa, protein_sequences)
    expected_result = np.zeros((3,4,0)) 
    np.testing.assert_array_equal(encoded_sequences, expected_result)
    
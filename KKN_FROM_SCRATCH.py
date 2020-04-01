#      KNN Algorithm from Scratch by: Charaf Eddine BENARAB


from math import sqrt
import csv
import heapq
from plot_data import plot_data

#Extracting features and labels from A CSV FILE
def data_prep(csv_file):
   
    with open(csv_file) as csv_ready:
        csv_reader = csv.reader(csv_ready,\
            delimiter=',')    
        labels = []
        row_data = []
        data_set = []
        for row in csv_reader:
            labels.append(row[len(row)-1])   
            row_data = row[:len(row)-1]
            row_data = list(map(float,row_data))
            data_set.append(row_data)  
    return data_set, labels

# Assuming only Euclidean_distance for its Popularity 

def euclidean_distance(dp1, dp2):
    dist = 0
    square_dist = 0
    for i in range(len(dp1)):
        square_dist += (dp1[i]-dp2[i])**2
    dist = sqrt(square_dist)
    return dist 

#Finding Neighbours

def set_elems_dists(data_set,labels, _input):
    distances  = []
    data_elements = []
    data_dict = dict()
    id2label = dict()
    #Storing Data elements and their Distance from Input element
    for _id, _data in enumerate(data_set):

        id2label[_id] = labels[_id]
        distances.append(euclidean_distance(_data, _input))
        data_elements.append(_id)
        data_dict[euclidean_distance(_data, _input)] = _id
        
    return distances, data_elements, data_dict,id2label

#Finiding the nearest points to our input 
def find_neighbours(data_elements,distances,data_dict,k,id2label):
    neighbours_ids = []
    neighbours_labels = []
    min_dists = heapq.nsmallest(k,distances)
    for dist in min_dists:
        neighbours_ids.append(data_dict[dist])
        neighbours_labels.append(id2label[data_dict[dist]])
    
    return neighbours_ids, neighbours_labels

#FUnction for finding element with majority in a list 
def majority_element(num_list):
        idx, ctr = 0, 1
        
        for i in range(1, len(num_list)):
            if num_list[idx] == num_list[i]:
                ctr += 1
            else:
                ctr -= 1
                if ctr == 0:
                    idx = i
                    ctr = 1
        
        return num_list[idx]

def KNN_algorithm(data_x, data_y,_input,k):
    
    distances, data_elements, data_dict,\
                        id2label = set_elems_dists(data_x,\
                            data_y,_input)
    _,neighbours_labels = find_neighbours\
        (data_elements, distances,data_dict,k,id2label)
    
    return majority_element(neighbours_labels) 

def main():

    data_x, data_y = data_prep('iris.csv')
    plot_data(data_x)
    #Getting input from user 
    while True:
        _input = []
        try:
            for i in range(4):
                feature = float(input("Enter feature Number {}:  \n".\
                    format(i+1)))
                _input.append(feature)

            k = int(input("Enter number of Neighbors:   "))
            
            category = KNN_algorithm(data_x=data_x,data_y=data_y,\
                _input=_input,k=k)
            
            print(category)
        except Exception as e:
            print(e)
            break

if __name__ == "__main__":
    main()


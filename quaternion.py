import numpy as np
import math
###################### Quaternion Related Codes ###########################
# Normalize the body structure by converting the joints to 
# quaternions and then converting back to joint coordinates
# using a standard bone "length". The hip of the skeleton is
# translated to the origin is "Shift" is set to True.
# Note: bone length can be read from Data/body_joint_length.pkl
# Note: edges is a list of bones which can be read from
# Data/KinectSkeleton.tree
def normbodystruct(X,length,edges,Shift=True):
    if Shift:
        # Shift the position to the hip
        X[:,0::3] -= np.reshape(X[:,0],(np.shape(X)[0],1))
        X[:,1::3] -= np.reshape(X[:,1],(np.shape(X)[0],1))
        X[:,2::3] -= np.reshape(X[:,2],(np.shape(X)[0],1))
    # Calculate quaternions
    quaternion = joint2q(X,edges)
    # Convert back to coordinate with preset lengths
    coord = q2joint(quaternion,length,edges)
    return coord

# Return length of all the bones from a skeleton frame
def bonelen(X_0,edges):
    length = []
    for anedge in edges:
        X1 = X_0[anedge[0]*3]
        Y1 = X_0[anedge[0]*3+1]
        Z1 = X_0[anedge[0]*3+2]
        X2 = X_0[anedge[1]*3]
        Y2 = X_0[anedge[1]*3+1]
        Z2 = X_0[anedge[1]*3+2]
        length.append(math.sqrt((X1-X2)**2+(Y1-Y2)**2+(Z1-Z2)**2))
    return length
    
# Given the joint coordinates and the edges list, \
# returns the corresponding quaternion array
def joint2q(X,edges):
    quaternion_array = np.zeros((len(X),76))
    for m in range (len(edges)):
        v1,v2 = calvec(m,X,edges)
        if m == 0:
            origin_unit_vector = v1
        quaternion = calcq(v1,v2)
        quaternion_array[:,m*4] = quaternion[:,0]
        quaternion_array[:,m*4+1] = quaternion[:,1]
        quaternion_array[:,m*4+2] = quaternion[:,2]
        quaternion_array[:,m*4+3] = quaternion[:,3]
    return quaternion_array

# Convert from quaternions to joints
def q2joint(Q,length,edges):
    M = len(Q)
    joints = np.zeros((M,60))
    quaternion_array = np.zeros((19,M,4))
    coordinates = np.zeros((M,60))
    for n in range(19):
        quaternion_array[n,0:M,0] = Q[0:M,n*4]
        quaternion_array[n,0:M,1] = Q[0:M,n*4+1]
        quaternion_array[n,0:M,2] = Q[0:M,n*4+2]
        quaternion_array[n,0:M,3] = Q[0:M,n*4+3]
    coordinates = recov(quaternion_array,length,edges)
    joints[0:M,:] = coordinates[0:M,:]
    return joints

# Returns the joint coordinates for a specific bone# (edgeid)
# v1 and v2 are two vectors representing the normalized
# coordinates for the corresponding joint
def calvec(edgeid, X, edges):
    v1 = np.zeros((len(X),3))
    v2 = np.zeros((len(X),3))
    # Parent Vector
    if edges[edgeid][0] == 0:
        v1 = np.tile(np.array([0,-1,0]),(len(X),1))
    else:
        parentid = 0
        X1 = X[:,edges[edgeid][0]*3]
        Y1 = X[:,edges[edgeid][0]*3+1]
        Z1 = X[:,edges[edgeid][0]*3+2]
        for n in range (0,19):
            if edges[n][1] == edges[edgeid][0]:
                parentid = n
        X2 = X[:,edges[parentid][0]*3]
        Y2 = X[:,edges[parentid][0]*3+1]
        Z2 = X[:,edges[parentid][0]*3+2]
        vector_length = np.sqrt((X1-X2)**2+(Y1-Y2)**2+(Z1-Z2)**2)
        v1 = np.column_stack(((X1-X2)/vector_length,(Y1-Y2)/vector_length,(Z1-Z2)/vector_length))

    # Current Vector
    X3 = X[:,edges[edgeid][1]*3]
    Y3 = X[:,edges[edgeid][1]*3+1]
    Z3 = X[:,edges[edgeid][1]*3+2]
    X4 = X[:,edges[edgeid][0]*3]
    Y4 = X[:,edges[edgeid][0]*3+1]
    Z4 = X[:,edges[edgeid][0]*3+2]
    vector_length = np.sqrt((X3-X4)**2+(Y3-Y4)**2+(Z3-Z4)**2)

    v2 = np.column_stack(((X3-X4)/vector_length,(Y3-Y4)/vector_length,(Z3-Z4)/vector_length))
    return np.nan_to_num(v1),np.nan_to_num(v2)

# Returns the original coordinates
# Rotvec used
def recov(quaternion, length, edges):
    # Unit vectors for rotvec
    unit_vector_array = np.zeros((19,len(quaternion[0]),3))
    # Vectors to find the coodinates
    vector_array = np.zeros((19,len(quaternion[0]),3))

    coodinate = np.zeros((len(quaternion[0]),60))
    origin_unit_vector = np.tile(np.array([0,-1,0]),(len(quaternion[0]),1))
    coodinate[:,0] = coodinate[:,1] = coodinate[:,2] = 0
    # Edges
    for m in range(0,len(quaternion)):
        # the Root vector from the origin
        if edges[m][0] == 0:
            unit_vector = rotvec(origin_unit_vector,quaternion[m])
            unit_vector_array[m] = unit_vector
            vector_array[m] = unit_vector_array[m] * length[m]
            coodinate[:,edges[m][1]*3] = coodinate[:,0] + vector_array[m,:,0]
            coodinate[:,edges[m][1]*3+1] = coodinate[:,1] + vector_array[m,:,1]
            coodinate[:,edges[m][1]*3+2] = coodinate[:,2] + vector_array[m,:,2]
        else:
            for x in range (0,19):
                # Find the parent edge
                if edges[m][0] == edges[x][1]:
                    parent_vector = x
            unit_vector = rotvec(unit_vector_array[parent_vector],quaternion[m])
            unit_vector_array[m] = unit_vector
            vector_array[m] = unit_vector * length[m]
            coodinate[:,edges[m][1]*3] = coodinate[:,edges[m][0]*3] + vector_array[m,:,0]
            coodinate[:,edges[m][1]*3+1] = coodinate[:,edges[m][0]*3+1] + vector_array[m,:,1]
            coodinate[:,edges[m][1]*3+2] = coodinate[:,edges[m][0]*3+2] + vector_array[m,:,2]
    return coodinate

                
# Returns the column numbers for a particular joint
def __jcols(jointid):
    return (2+jointid*3,3+jointid*3,4+jointid*3)

# Calculate the orientations (in Quaternion) from two unit bone vectors
# Returns the axis of orientation and sin(theta) where theta represents the 
# orientation angle with respect to the axis. 
# prevU represents the previous unit bone vector.
def calcq(v1,v2):
    costh = np.sum(v1*v2,axis=1)
    w = np.sqrt(0.5*(1.0+costh))[None].T
    u = np.nan_to_num(np.cross(v1,v2)/(2.0*w))
    return np.concatenate((w,u),axis=1)

# Calculate the conjugate of q
def __cnjq(q):
    q[:,1:]*=-1.0
    return q
# Calcualtes the product of two quaternions
def __quatprod(q1,q2):
    w = q1[:,0]*q2[:,0] - np.sum(q1[:,1:]*q2[:,1:],axis=1)
    u = q1[:,0][None].T*q2[:,1:] + q2[:,0][None].T*q1[:,1:] + np.cross(q1[:,1:],q2[:,1:])
    return np.concatenate((w[None].T,u),axis=1)
## Rotate a vector v with a quarternion q. This is a faster
## equivalent of q*v*congugate(q)
def rotvec(v,q):
    if len(v)!=len(q):
        raise ValueError('Length of the first dimension must match')
    if np.size(v,axis=1)<np.size(q,axis=1):
        v = np.concatenate((np.zeros((len(v),1)),v),axis=1)
    return __quatprod(__quatprod(q,v),__cnjq(q))[:,1:]
    
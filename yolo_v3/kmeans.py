"""
kmeans algorithm for anchors
Written by LI Shuai
"""
import numpy as np

class YOLO_Kmeans(object):
    def __init__(self,cluster_number):
        self.cluster_number=cluster_number
        self.filename='data/train.txt'

    def iou(self,boxes,clusters):
        """
        calculate the iou between boxes and clusters
        :param boxes:
        :param clusters:
        :return:
        """
        n=boxes.shape[0]
        k=self.cluster_number

        box_area=boxes[:,0]*boxes[:,1]
        box_area=box_area.repeat(k)
        box_area=np.reshape(box_area,(n,k))

        clusters_area=clusters[:,0]*clusters[:,1]
        clusters_area=np.tile(clusters_area,[1,n])
        cluster_area = np.reshape(clusters_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result
    def avg_iou(self,boxes,clusters):
        acc=np.mean([np.max(self.iou(boxes,clusters),axis=1)])
        return acc
    def kmeans(self,boxes,k,dist=np.median):
        box_number=boxes.shape[0]
        distances=np.empty((box_number,k))
        last_nearest=np.zeros((box_number,))
        np.random.seed()
        clusters=boxes[np.random.choice(box_number,k,replace=False)]
        while True:
            distances=1-self.iou(boxes,clusters)
            current_nearest=np.argmin(distances,axis=1)
            if (last_nearest==current_nearest).all():
                break
            for cluster in range(k):
                clusters[cluster]=dist(boxes[current_nearest==cluster],axis=0)
            last_nearest=current_nearest
        return clusters
    def main(self):
        traintxt=open(self.filename,'r')
        dataset=[]
        for line in traintxt:
            infos=line.split(" ")
            length=len(infos)
            for i in range(1,length):
                width=int(infos[i].split(',')[2])-int(infos[i].split(',')[0])
                height=int(infos[i].split(",")[3])-int(infos[i].split(",")[1])
                dataset.append([width,height])
        traintxt.close()
        dataset=np.array(dataset)
        result=self.kmeans(dataset,k=self.cluster_number)
        result=result[np.lexsort(result.T[0,None])]
        f=open('data/yolo_anchors.txt','w')
        row=np.shape(result)[0]
        for i in range(row):
            if i==0:
                x_y="%d,%d"%(result[i][0],result[i][1])
            else:
                x_y=",%d,%d"%(result[i][0],result[i][1])
            f.write(x_y)
        f.close()
        print("K anchors:\n {}".format(result))
        print('accuracy:{:.2f}%'.format(self.avg_iou(dataset,result)*100))











# import the necessary packages
import torch
from scipy.special import softmax
from collections import OrderedDict
import numpy as np
from sklearn.metrics.pairwise import cosine_distances


class ReidTracker:
    def __init__(
		self, maxDisappeared=100, n_init=1, max_cosine_dist=0.5,
        use_bisoftmax=False, instance_count_for_matching=10, momentum=0.9):

		# initialize
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

		# store the number of maximum consecutive frames to disappear
        self.maxDisappeared = maxDisappeared

        self.n_init = n_init
        self.status = OrderedDict()
        self.hits = OrderedDict()
        self.deregistered_obj = []
        self.emb_gallery_norm = OrderedDict()
        self.max_cosine_dist = max_cosine_dist
        self.use_bisoftmax = use_bisoftmax
        self.instance_count_for_matching = instance_count_for_matching
        # Set momentum to -1 to use simple average
        self.momentum = momentum

    def register(self, bbox, emb_norm):
        self.objects[self.nextObjectID] = bbox
        self.disappeared[self.nextObjectID] = 0
        self.hits[self.nextObjectID] = 1
        self.emb_gallery_norm[self.nextObjectID] = [emb_norm]
        self.status[self.nextObjectID] = "Tentative"
        if self.hits[self.nextObjectID] >= self.n_init:
            self.status[self.nextObjectID] = "Confirmed"
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.hits[objectID]

        if self.status[objectID] == "Confirmed":
            self.deregistered_obj.append(objectID)

        del self.status[objectID]
        del self.emb_gallery_norm[objectID]

    def _calc_cosine_dist(self, emb_gallery_norm, emb_det_norm):
        cosine_dist_mat = cosine_distances(emb_gallery_norm, emb_det_norm)
        if self.use_bisoftmax:
			# https://arxiv.org/pdf/2006.06664.pdf
            cosine_dist_mat1 = softmax(cosine_dist_mat, 0)
            cosine_dist_mat2 = softmax(cosine_dist_mat, 1)
            cosine_dist_mat = cosine_dist_mat1 + cosine_dist_mat2
        return cosine_dist_mat

    def update(self, bbox, emb_det_norm):
        if bbox is None or len(bbox) == 0:
			# if no detection, add disappearance count by 1
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # Deregister an obj
                if self.status[objectID] == "Confirmed" and self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
                elif self.status[objectID] == "Tentative" and self.disappeared[objectID] > self.maxDisappeared/10:
                    self.deregister(objectID)

            return {id_:bbox for id_, bbox in self.objects.items() if self.disappeared[id_] == 0}

        bbox = bbox[:, :4] if isinstance(bbox, np.ndarray) else bbox.cpu().numpy()[:, :4]

		# Register the detection if currently_tracked obj is 0
        if len(self.objects) == 0:
            for i in range(0, len(bbox)):
                self.register(bbox[i], emb_det_norm[i])

        # Perform matching
        else:
			# grab the set of object IDs
            objectIDs = list(self.objects.keys())

            # Get gallery emmbedding
            emb_gallery_norm = []
            for emb_gallery_norm_tmp in self.emb_gallery_norm.values():
                emb_gallery_norm_tmp = torch.stack(emb_gallery_norm_tmp, 0)
                emb_gallery_norm_tmp = emb_gallery_norm_tmp.mean(0, keepdim=True)
                emb_gallery_norm_tmp = torch.nn.functional.normalize(
                    emb_gallery_norm_tmp, dim=1, p=2
                    )
                emb_gallery_norm.append(emb_gallery_norm_tmp)
            emb_gallery_norm = torch.cat(emb_gallery_norm, 0)

            # Compute cosine distance between embeddings
            cosine_dist = self._calc_cosine_dist(emb_gallery_norm, emb_det_norm)

            rows = cosine_dist.min(axis=1).argsort()
            cols = cosine_dist.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()

			# loop over the combination of the (row, column) index
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                # Set as non-match if dist is large
                if cosine_dist[row, col] > self.max_cosine_dist:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = bbox[col]
                self.hits[objectID] += 1
                self.disappeared[objectID] = 0
                if self.momentum == -1:
                    # Simple moving average
                    self.emb_gallery_norm[objectID].append(emb_det_norm[col])
                    self.emb_gallery_norm[objectID] = self.emb_gallery_norm[objectID][-self.instance_count_for_matching:]
                else:
                    self.emb_gallery_norm[objectID] =\
                        [self.momentum*self.emb_gallery_norm[objectID][0] +\
                            (1-self.momentum)*emb_det_norm[col]]

                if self.status[objectID] == "Tentative" and self.hits[objectID] >= self.n_init:
                    self.status[objectID] = "Confirmed"

				# indicate that we have examined each of the row and
				# column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

			# compute both the row and column index we have NOT yet
			# examined
            unusedRows = set(range(0, cosine_dist.shape[0])).difference(usedRows)
            unusedCols = set(range(0, cosine_dist.shape[1])).difference(usedCols)

			# if tracked id is more than detected obj
            if cosine_dist.shape[0] >= cosine_dist.shape[1]:
                for row in unusedRows:
					# grab the object ID for the corresponding row
					# index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

					# check to see if the number of consecutive
					# frames the object has been marked "disappeared"
					# for warrants deregistering the object
                    if self.status[objectID] == "Confirmed" and self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
                    elif self.status[objectID] == "Tentative" and self.disappeared[objectID] > self.maxDisappeared/10:
                        self.deregister(objectID)

            else:
                # if detected_obj is more than tracked_id
                for col in unusedCols:
                    self.register(bbox[col], emb_det_norm[col])

		# return the set of trackable objects
        return {id_: bbox for id_, bbox in self.objects.items() if self.disappeared[id_] == 0}

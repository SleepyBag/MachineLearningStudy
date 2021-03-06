__author__ = 'trimi'


import numpy as np


class loadMovieData:
    def read_training_data(self, path):
        with open(path, 'rb') as f:
            matrix = []
            userItems = {}
            itemUsers = {}
            max_item = []
            timestamps = []
            b = 0

            for line in f:

                row = []
                if b % 10000 == 0:
                    print('b  = ', b)

                a = line.split(b'\t')

                user = int(a[0])
                item = int(a[1])
                rating = float(a[2])
                time_ = int(a[3])

                row.append(user)
                row.append(item)
                row.append(rating)
                row.append(time_)

                matrix.append(row)
                max_item.append(item)
                timestamps.append(time_)

                # pos events per user
                if user not in userItems:
                    userItems[user] = [(item, rating, time_)]
                else:
                    if item not in userItems[user]:
                        userItems[user].append((item, rating, time_))

                # items rated by users
                if item not in itemUsers:
                    itemUsers[item] = [(user, rating, time_)]
                else:
                    if user not in itemUsers[item]:
                        itemUsers[item].append((user, rating, time_))

                b += 1
            print('#pos_events = ', b)
            min_timestamp = min(timestamps)
            max_timestamp = max(timestamps)

            print('max item id = ', max(max_item))
            return matrix, userItems, itemUsers, min_timestamp, max_timestamp

    def num_of_days(self, min_timestamp, max_timestamp):
        nDays = int((max_timestamp - min_timestamp)/86400)

        return nDays

    def cal_day(self, timestamp, num_of_days, min_timestamp):
        day_ind = np.minimum(
            num_of_days - 1, int((timestamp - min_timestamp)/86400))

        return day_ind

    def timestamp_to_day(self, matrix, n_days, min_timestamp):

        userItems = {}
        itemUsers = {}

        for i in range(len(matrix)):
            user = matrix[i][0]
            item = matrix[i][1]
            rating = matrix[i][2]
            timestamp = matrix[i][3]

            day_ind = self.cal_day(timestamp, n_days, min_timestamp)

            # pos events per user
            if user not in userItems:
                userItems[user] = [(item, rating, day_ind)]
            else:
                if item not in userItems[user]:
                    userItems[user].append((item, rating, day_ind))

            # items rated by users
            if item not in itemUsers:
                itemUsers[item] = [(user, rating, day_ind)]
            else:
                if user not in itemUsers[item]:
                    itemUsers[item].append((user, rating, day_ind))

        return userItems, itemUsers

    def main(self, file_name):
        matrix, userItems, itemUsers, min_timestamp, max_timestamp = \
            self.read_training_data(file_name)
        num_days = self.num_of_days(min_timestamp, max_timestamp)
        new_userItems, new_itemUsers = self.timestamp_to_day(
            matrix, num_days, min_timestamp)

        nUsers = len(new_userItems)
        nItems = 1683

        return new_userItems, nUsers, nItems, num_days, min_timestamp

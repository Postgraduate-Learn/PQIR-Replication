class Post:

    def __init__(self, row):
        self.id = row.get('Id')
        self.post_type_id = row.get('PostTypeId')

        if row.get('ParentID') is not None:
            self.parent_id = row.get('ParentID')
        else:
            self.parent_id = ''

        if row.get('AcceptedAnswerId') is not None:
            self.accepted_answer_id = row.get('AcceptedAnswerId')
        else:
            self.accepted_answer_id = ''

        if row.get('CreationDate') is not None:
            self.creation_date = row.get('CreationDate')
        else:
            self.creation_date = ''

        if row.get('Score') is not None:
            self.score = row.get('Score')
        else:
            self.score = ''

        if row.get('ViewCount') is not None:
            self.view_count = row.get('ViewCount')
        else:
            self.view_count = ''

        if row.get('Body') is not None:
            self.body = row.get('Body')
        else:
            self.body = ''

        if row.get('LastActivityDate') is not None:
            self.last_activity_date = row.get('LastActivityDate')
        else:
            self.last_activity_date = ''

        if row.get('CloseDate') is not None:
            self.close_date = row.get('CloseDate')
        else:
            self.close_date = ''

        if row.get('Title') is not None:
            self.title = row.get('Title')
        else:
            self.title = ''

        if row.get('Tags') is not None:
            self.tags = row.get('Tags')
        else:
            self.tags = ''

        if row.get('AnswerCount') is not None:
            self.answer_count = row.get('AnswerCount')
        else:
            self.answer_count = ''

        if row.get('CommentCount') is not None:
            self.comment_count = row.get('CommentCount')
        else:
            self.comment_count = ''

        if row.get('FavoriteCount') is not None:
            self.favorite_count = row.get('FavoriteCount')
        else:
            self.favorite_count = ''

        if row.get('OwnerUserId') is not None:
            self.owner_user_id = row.get('OwnerUserId')
        else:
            self.owner_user_id = ''

    def to_dict_s(self):
        return {
            'Id': self.id,
            'PostTypeId': self.post_type_id,
            'ParentID': self.parent_id,
            'AcceptedAnswerId': self.accepted_answer_id,
            'CreationDate': self.creation_date,
            'Score': self.score,
            'ViewCount': self.view_count,
            'Body': self.body,
            'CloseDate': self.close_date,
            'LastActivityDate': self.last_activity_date,
            'Title': self.title,
            'Tags': self.tags,
            'AnswerCount': self.answer_count,
            'CommentCount': self.comment_count,
            'FavoriteCount': self.favorite_count,
            'OwnerUserId': self.owner_user_id
        }

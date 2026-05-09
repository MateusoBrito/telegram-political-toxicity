The final fields for each TSV file are as follows:

'id' -> message ID
'user_id' -> user ID (NaN if not available, such as in channels)
'text' -> None if empty
'timestamp' -> I used timestamp instead of date to save space
'bot_flag' -> 1 if the 'user_id' is in the bot list from 'bot_info' in the metadata
'bot_flag'
'via_bot_id' -> ID of the inline bot that generated the message (NaN if not available)
'via_business_bot_id'
'reply_to_msg_id' -> ID of the message being replied to if the message is a reply; NaN otherwise
'fwd_flag' -> 1 if the message was forwarded
'fwd_from_id' -> ID of the channel/group/user from which the message was forwarded
'media_type' -> category of non-text media in the message
'views' -> number of views for the message
'forwards' -> number of times the message was forwarded
'replies' -> number of replies the message received
'reactions' -> total number of reactions
'reaction_json' -> JSON format (can be loaded with json.loads()) where the key is the emoticon and the value is the number of times it was used. This could be useful for sentiment analysis to understand how the audience received the message.
 
Some practical tips: 1) Although it should have already been avoided, I would always do a df.drop_duplicates(subset=['id']) just in case. 2) Sort the dfs  by timestamp or by 'id' because I am not sure if the original data was sorted.

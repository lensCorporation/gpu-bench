default_list = 'main'
watchlists = {}
conflict_solution = "latest" # solution when key conflicts, one of followings ["oldest", "latest", "exception"]

def set_default(listname):
	default_list = listname

def list_all_lists():
	return watchlists.keys()


def insert(key, var, listname=None):
	# Get wathclist, if not exist, create it
	if listname is None:
		listname = default_list
	if not listname in watchlists:
		watchlists[listname] = {}
	watchlist = watchlists[listname]

	# Insert the variable into the list
	assert type(key) is str, "the key for tfwatcher must be an string!"
	if key in watchlist.keys():
		if conflict_solution == "oldest":
			pass
		elif conflict_solution == "latest":
			watchlist[key] = var
		elif conflict_solution == "exception":
			raise KeyError("Trying to insert an variable with key {} \
				when it is already in the watchlist of tfwatcher!".format(key))
		else:
			raise ValueError("Unknown conflict_solution in tfwatcher: {}".format(conflict_solution))
	else:
		watchlist[key] = var

def get_watchlist(listname=None):
	if listname is None:
		listname = default_list
	return watchlists[listname]

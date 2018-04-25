import csv
import pandas as pd


class Relation:

    str_to_type = {'int': int, 'float': float, 'str': str}
    type_to_str = {int: 'int', float: 'float', str: 'str'}

    def __init__(self, name, schema):
        """
        Instantiates an object of the Relation class

        :param name: name of the relation or expression the relation object was built from
        :param schema: list of tuples containing attributes names and types
        """
        # derive attributes and domains from schema
        attributes, domains = Relation._parse_schema(schema)
        # set class variables
        self.name = name  # name of the relation or expression the relation object was built from
        self.attributes = attributes  # tuple of attribute names
        self.domains = domains  # tuple of attribute types
        self.tuples = set()  # this ensures not having duplicates

    def add_tuple(self, tup):
        """
        Adds the tuple tup to the relation

        :param tup: tuple to be added to the relation
        """
        # check that tuple is valid
        assert isinstance(tup, tuple)  # tuple should be of type tuple
        assert len(tup) == len(self.domains)  # tuple should have correct amount of attributes
        assert all(map(lambda x: isinstance(x[0], x[1]), zip(tup, self.domains)))  # types of all attributes must match relation
        # add tuple
        self.tuples.add(tup)
    
    def set_name(self, name):
        """
        Changes the name of the relation object to the name provided
        
        :param: name: the new name for the relation
        """    
        self.name = name
    
    def card(self):
        """
        Computes the cardinality of the relation, i.e. the amount of tuples contained

        :return: the amount of tuples in the relation
        """
        return len(self.tuples)

    def print_schema(self):
        """
        Prints the schema of the relation
        """
        # here string representation of a relation is its schema
        print(str(self))

    def print_table(self):
        """
        Prints the relation and its tuples in a tabular layout
        """
        # calculate column width for printing
        col_width = self._get_col_width()
        # relation name bold
        target = '-'*len(self.name) + '\n' \
                 '\033[1m' + self.name + '\033[0m \n'
        # attribute names bold
        target += '-'*max(len(self.name), col_width*len(self.attributes)) + '\n' \
                  '\033[1m' + ''.join(attr_name.ljust(col_width) for attr_name in self.attributes) + '\033[0m \n'
        target += '-'*max(len(self.name), col_width*len(self.attributes)) + '\n'
        # tuples
        for tup in self.tuples:
            target += ''.join(str(attr_val).ljust(col_width) for attr_val in tup) + '\n'
        # print target
        print(target)

    def print_set(self):
        """
        Prints the relation and its tuples in set notation
        """
        target = self.__str__() + '\n{\n'
        for tup in self.tuples:
            target += '\t(' + ', '.join(str(attr) for attr in tup) + '),\n'
        target = target.rstrip("\n").rstrip(",")
        target += '\n}'
        print(target)

    def print_latex(self):
        """
        Prints LaTeX code for the relation in tabular layout
        """
        num_cols = len(self.attributes)
        latex = '\\begin{tabular}{'+('l'*num_cols)+'}\n'\
                + '\t'+' & '.join(attr for attr in self.attributes)+' \\\\\n' \
                + '\t\\hline\n'
        for tup in self.tuples:
            latex += '\t'+' & '.join(str(attr) for attr in tup)+' \\\\\n'
        latex += '\\end{tabular}'
        print(latex)

    def to_DataFrame(self):
        """
        Converts the relation into a pandas DataFrame

        :return: DataFrame representation of the relation
        """
        df = pd.DataFrame(list(self.tuples), columns=self.attributes)
        df.set_index(list(self.attributes), inplace=True)
        return df

    # internal helper functions
    @staticmethod
    def _parse_schema(schema):
        """
        Splits up the schema from the init method into attributes and domains

        :param schema: the schema provided to the init method by the user
        :return attributes: tuple of attribute names
        :return domains: tuple of domain types
        """
        # split schema into attributes and domains
        attributes = tuple([x[0] for x in schema])  # first position is attribute name
        domains = tuple([x[1] for x in schema])  # second position is domain type
        # check that attribute names and domain types are valid
        assert all(map(lambda x: x.isidentifier(), attributes))  # all elements of attributes should be an identifier
        assert len(set(attributes)) == len(attributes)  # each attribute name should be unique
        assert all(map(lambda x: isinstance(x, type), domains))  # all elements of domains should be types
        # return attributes and domains
        return attributes, domains

    def _get_col_width(self):
        """
        Computes the maximum column width required to represent the relation in tabular layout

        :return: the maximum column width required
        """
        attr_name_width = max(len(attr_name) for attr_name in self.attributes)
        attr_val_width = max((len(str(attr_val)) for tup in self.tuples for attr_val in tup), default=0)
        return max(attr_name_width, attr_val_width) + 2  # padding

    # external helper functions
    def has_attribute(self, attribute):
        """
        Determines if the relation has a given attribute

        :param attribute: the attribute name to be tested
        :return: True if the relation has a attibute with the given name, false otherwise
        """
        return attribute in self.attributes

    def get_attribute_domain(self, attribute):
        """
        Determines the domain of a given attribute

        :param attribute: the name of the attribute
        :return: the domain of the given attribute
        """
        # integrity checks
        assert self.has_attribute(attribute)  # relation should have the attribute
        # return attr domain
        return self.domains[self.attributes.index(attribute)]

    def get_attribute_index(self, attribute):
        """
        Determines the position of a given attribute in the tuples of a relation

        :param attribute: the name of the attribute
        :return: the position of the attribute in a tuple of the relation
        """
        # integrity checks
        assert self.has_attribute(attribute)  # relation should have the attribute
        # return index
        return self.attributes.index(attribute)

    def __str__(self):
        """
        Computes a string representation of the relation object, here the schema

        :return: string representation of the object
        """
        rel = '[{}]'.format(self.name)
        attrs = ','.join(
            [' {}:{}'.format(self.attributes[i], Relation.type_to_str[self.domains[i]])
             for i in range(len(self.attributes))])
        return '{} : {{[{} ]}}'.format(rel, attrs)

    def __len__(self):
        """
        Computes the amaount of tuples in the given relation

        :return: the amount of tuples contained in the relation
        """
        return len(self.tuples)

    def __eq__(self, other):
        """
        Computes whether two relations are equal, i.e. they contain the same tuples

        Note: This does not consider attribute names.

        :param other: the relation to compare to self
        :return: True if the relations are equal
        """
        return self.tuples == other.tuples


###############
# CSV PARSING #
###############


def load_csv(path, name, delimiter=',', quotechar='"'):
    """
    Loads a .csv File into a new relation object

    :param path: the path to the .csv file
    :param name: the name of the relation
    :param delimiter: the delimiter used in the .csv file
    :param quotechar: the char used for quotes in the .csv file
    :return: a new relation with the data from the specified .csv file
    """
    # load csv into pandas df
    with open(path) as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter, quotechar=quotechar)
        # extract header
        attributes = next(reader)
        # build attribute list and add tuples
        domains = list()
        relation = None
        for i, row in enumerate(reader):
            # build attribute list based on first row
            if i == 0:
                domains = get_domains(row)
                schema = build_schema(attributes, domains)
                # build relation
                relation = Relation(name, schema)
            # insert tuples
            tup = build_tuple(row, domains)
            relation.add_tuple(tup)
        # return relation
        return relation


def get_domains(row):
    """
    Extracts the domain types from a row in a .csv file

    :param row: a list of strings from the .csv reader
    :return: a list of domain types for the strings in row
    """
    # integrity checks
    assert len(row) > 0, 'Row does not contain any data'
    # build domains list
    domains = list()
    for attr in row:
        if attr.isdigit():
            domains.append(int)
        elif isfloat(attr):
            domains.append(float)
        else:
            domains.append(str)
    return domains


def build_tuple(row, domains):
    """
    Builds a tuple to be inserted into the relation from the row and domains

    :param row: a list of strings representing the values in the tuple
    :param domains: a list of domain types
    :return: a tuple of accordingly typed values from the row
    """
    # integrity checks
    assert len(row) == len(domains)  # The length of the row and domains should match
    assert all(map(lambda x: isinstance(x, type), domains))  # All domains need to be types
    # build tuple
    return tuple(dom(attr) for attr, dom in zip(row, domains))


def isfloat(input):
    """
    Determines whether a string can be interpreted as float

    :param input: the input string
    :return: True if the input can be interpreted as float, False otherwise
    """
    try:
        float(input)
        return True
    except ValueError:
        return False

    
#############
# OPERATORS #
#############


# PROJECTION
def projection(relation, attributes):
    """
    Performs projection for a list of attributes on a given relation

    :param relation: a relation object
    :param attributes: a list of attributes names
    :return: the relation projected onto the specified attributes
    """
    # integrity checks
    assert all(map(relation.has_attribute, attributes))  # the relation should have all specified attributes
    # create empty new relation
    new_name = 'π_{'+','.join(attributes)+'}('+relation.name+')'
    new_schema = build_schema(attributes, [*map(relation.get_attribute_domain, attributes)])
    new_relation = Relation(new_name, new_schema)
    # add tuples to new relation
    attr_indexes = [*map(relation.get_attribute_index, attributes)]
    for tup in relation.tuples:
        new_tup = tuple(tup[i] for i in attr_indexes)
        new_relation.add_tuple(new_tup)  # automatically eliminates duplicates
    return new_relation


# SELECTION
def selection(relation, condition):
    """
    Performs selection for a condition on a given relation

    :param relation: the relation object
    :param condition: a condition of tuples that should be kept
    :return: a new relation only containing tuples that match the condition
    """
    # create empty new relation
    new_name = 'σ_{'+str(condition)+'}('+relation.name+')'
    new_schema = build_schema(relation.attributes, relation.domains)
    new_relation = Relation(new_name, new_schema)
    # add tuples to new relation
    for tup in relation.tuples:
        if eval(condition, locals_dict(tup, relation.attributes)):  # checks whether tuple fulfills condition
            new_relation.add_tuple(tup)  # implicitly handles duplicate elimination
    return new_relation


def locals_dict(tup, attributes):
    """
    Builds a dictionary mapping the attribute names to the corresponding value in the tuple

    :param tup: the tuple
    :param attributes: the attribute names
    :return: a dictionary mapping attribute names to tuple values
    """
    # integrity checks
    assert len(tup) == len(attributes)  # tuple and attributes should have the same length
    # build dictionary
    return {attr: val for attr, val in zip(attributes, tup)}


# UNION
def union(relation1, relation2):
    """
    Computes the set union of two relations

    :param relation1: the first relation
    :param relation2:  the second relation
    :return: a new relation representing the set union of the given relations
    """
    # integrity checks
    assert relation1.attributes == relation2.attributes  # the schema of both relations has to be identical
    assert relation1.domains == relation2.domains  # # the schema of both relations has to be identical
    # create empty new relation
    new_name = '('+relation1.name+') ∪ ('+relation2.name+')'
    new_schema = build_schema(relation1.attributes, relation1.domains)
    new_relation = Relation(new_name, new_schema)
    # add tuples to new relation
    for tup in relation1.tuples | relation2.tuples:
        new_relation.add_tuple(tup)
    return new_relation


# DIFFERENCE
def difference(relation1, relation2):
    """
    Computes the set difference of two relations

    :param relation1: the first relation
    :param relation2:  the second relation
    :return: a new relation representing the set difference of the given relations
    """
    # integrity checks
    assert relation1.attributes == relation2.attributes  # the schema of both relations has to be identical
    assert relation1.domains == relation2.domains  # # the schema of both relations has to be identical
    # create empty new relation
    new_name = '('+relation1.name+') - ('+relation2.name+')'
    new_schema = build_schema(relation1.attributes, relation1.domains)
    new_relation = Relation(new_name, new_schema)
    # add tuples to new relation
    for tup in relation1.tuples - relation2.tuples:
        new_relation.add_tuple(tup)
    return new_relation


# INTERSECTION
def intersection(relation1, relation2):
    """
    Computes the intersection of two relations

    :param relation1: the first relation
    :param relation2:  the second relation
    :return: a new relation representing the intersection of the given relations
    """
    # integrity checks
    assert relation1.attributes == relation2.attributes  # the schema of both relations has to be identical
    assert relation1.domains == relation2.domains  # # the schema of both relations has to be identical
    # create empty new relation
    new_name = '(' + relation1.name + ') ∩ (' + relation2.name + ')'
    new_schema = build_schema(relation1.attributes, relation1.domains)
    new_relation = Relation(new_name, new_schema)
    # add tuples to new relation
    for tup in relation1.tuples & relation2.tuples:
        new_relation.add_tuple(tup)
    return new_relation


# CARTESIAN PRODUCT
def cartesian_product(relation1, relation2):
    """
    Computes the cartesian product of two relations
    Note: In favor of simplicity, we do not allow equally named attributes here. This avoids having to use the dot
    notation (relation.attribute) in order to access attributes with the same name

    :param relation1: the first relation
    :param relation2:  the second relation
    :return: a new relation computed by applying the cartesian product
    """
    # integrity checks
    assert len(set(relation1.attributes) & set(relation2.attributes)) == 0  # we do not allow equally named attributes here
    # create empty new relation
    new_name = '('+relation1.name+') × ('+relation2.name+')'
    new_schema = build_schema(relation1.attributes+relation2.attributes, relation1.domains+relation2.domains)
    new_relation = Relation(new_name, new_schema)
    # insert cartesian product of tuples
    for tup1 in relation1.tuples:
        for tup2 in relation2.tuples:
            new_relation.add_tuple(tup1+tup2)
    return new_relation


# RENAMING (RELATION)
def renaming_relation(relation, new_name):
    """
    Performs renaming on the relation
    Note: that in our case renaming a relation isn't really powerful as we do not allow equal attribute names and, thus,
    dot-access (relation.attribute) is never required.

    :param relation: the relation object
    :param name: the new name of the relation
    :return: a new relation object with the same schema but new name
    """
    assert new_name.isidentifier()  # the name should be an identifier
    # build new empty relation and parse changes
    new_schema = build_schema(relation.attributes, relation.domains)  # schema is left untouched
    new_relation = Relation(new_name, new_schema)
    # add all existing tuples
    for tup in relation.tuples:
        new_relation.add_tuple(tup)
    return new_relation


# RENAMING (ATTRIBUTES)
def renaming_attributes(relation, changes):
    """
    Performs renaming on the attributes of a relation

    :param relation: the relation object
    :param changes: a list of name changes of the form 'new_name<-old_name'
    :return: a new relation object with the attribute name changes applied
    """
    # build new empty relation and parse changes
    new_name = 'ρ_{'+','.join(changes)+'}('+relation.name+')'
    new_attributes = relation.attributes
    # apply each change to the attribute names
    for expr in changes:
        new_attributes = parse_attribute_rename(expr, new_attributes)
    new_schema = build_schema(new_attributes, relation.domains)
    new_relation = Relation(new_name, new_schema)
    # insert old tuples
    for tup in relation.tuples:
        new_relation.add_tuple(tup)
    return new_relation


def parse_attribute_rename(expr, attributes):
    """
    Apply the name change to the attributes

    :param expr: expression describing the change of the form 'new_name<-old_name'
    :param attributes: the list of attributes that the change is to be applied to
    :return: a list of attributes with the change applied
    """
    split = expr.split('<-')
    # integrity checks
    assert len(split) == 2  # after the split there should just be an old name and a new one
    assert all(map(lambda x: x.isidentifier(), split))  # the attribute names should be identifiers
    # parse expression
    old_attr = split[1]
    new_attr = split[0]
    tmp_attributes = list(attributes)  # tuple do not allow item assignment
    for i, attr in enumerate(tmp_attributes):
        if attr == old_attr:
            tmp_attributes[i] = new_attr
            return tuple(tmp_attributes)
    raise ValueError


# HELPERS
def build_schema(attributes, domains):
    """
    Builds the schema needed to initialize a new relation object

    :param attributes: a list of attribute names
    :param domains: a list of domain types
    :return: a list of tuples consisting of attribute name and domain to initialize a new relation object
    """
    # integrity checks
    assert len(attributes) == len(domains)  # Length of attributes and domains should match
    assert all(map(lambda x: isinstance(x, str), attributes))  # All attributes need to be strings
    assert all(map(lambda x: isinstance(x, type), domains))  # All domains need to be types
    # build attribute list
    return [*map(lambda x: (x[0], x[1]), zip(attributes, domains))]

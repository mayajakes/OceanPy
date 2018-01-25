
class createNetCDF(object):

    def __init__(self, output_file):
        self.output_file = output_file
        self.dataset = Dataset(output_file, 'w')

        # Add global attributes
        self.dataset.Conventions = 'CF-1.6'
        self.dataset.Metadata_Conventions = 'Unidata Dataset Discovery v1.0'

    def add_dims(self, dims):
        ''' Create dimensions of variables to store in NetCDF '''
        for dimname, size in dims.items():
            self.dataset.createDimension(dimname=dimname, size=size)


    def add_glob_attr(self, glob_attr):
        ''' Add global attributes '''

        for key in glob_attr.keys():
            setattr(self.dataset, key, glob_attr[key])

    def create_vars(self, vars):

        import xml.etree.ElementTree as ET
        from urllib.request import urlopen
        # open cf conventions standard names xml file
        root = ET.parse(
            urlopen('http://cfconventions.org/Data/cf-standard-names/current/src/cf-standard-name-table.xml'))

        # check if var_name is in standard names xml file and if so save in dict with according unit and description
        standard_names = {}
        for child in root.findall('entry'):
            standard_names[child.get('id')] = (child.findtext('canonical_units'), child.findtext('description'))

        for varname, values in vars.items():
            datatype, dimensions, data = values

            # create variables
            var = self.dataset.createVariable(varname, datatype, dimensions)

            # add variable attributes
            if varname.split()[-1] in standard_names.keys():
                var.standard_name = varname.split()[-1]
                var.units = standard_names[varname.split()[-1]][0]
            var.long_name = varname
            # var.description = standard_names[varname.split()[-1]][1]

            # add data to variables
            var[:] = data

    def close(self):
        self.dataset.close()

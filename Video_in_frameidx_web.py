# from pyvi import ViUtils
# t=ViUtils.add_accents(u'duoc')
# print(t)
from PIL import Image
import requests
import re
import base64
from io import BytesIO
# url = 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxMTEhUQExIWFhUVFxcXFRYVFRUVFhcWFhUXFhcVFRUYHSghGBsmHRgVITEhJSkrLi4uGCAzODMtNygtMCsBCgoKDg0OFw8QFSsdHR0tLSsrLSstLSsrKy0rLSsrLS0tKy0tLS0rKy0tLS0tLS0tLSstLS0tKy0tLSsrKy0tK//AABEIALgBEgMBIgACEQEDEQH/xAAcAAEAAgMBAQEAAAAAAAAAAAAABQYDBAcBAgj/xABGEAACAQIDBQUEBwMLAwUAAAABAgADEQQhMQUGEkFREyJhcYEHMpGhI0JSYnKCsRSiwRUzQ1Njc5KywtHwFpPxJDVUlKP/xAAWAQEBAQAAAAAAAAAAAAAAAAAAAQL/xAAcEQEBAQACAwEAAAAAAAAAAAAAAREhMVFhcRL/2gAMAwEAAhEDEQA/AO4xEQEREBERAREQEREBERARK3vDvrhsLSqVOI1DTFiKYLLx6LTaoBwKxNhYm+ek5vQ3vxmOuyJprxYp8PSW+gSnR+kcfeY5+EmjtkTjaNjemH/+zjz8+Oa+O3j2hhF7QJYDVkxFWvTH95RrXIXxU5RvpcdsiVXdjfihiKC1KrLRqe66M1lDDmrnLhIII8DLRTcMAwIIOhBuD5GVH1ERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQMWJxC00ao7BUUEszGwAGpJlB3g3maseyUMEPu0VJWrVH2qza0afPhyJGpHunD7W9svTNKihXJWqkNaxcMFpkg5EL3nsdSq9JE7DWkUAov2nHm9W5LOxzNyc9eUzebioPe/BsaaPUqFgtWkGpr3aFOlxi/CngbZ5eUruzaYoF2BN197ve6UbgYfhIKkTreNwVBKZp1bO1RbCkLcTg5ach945Cc62js1Oy7Cj31B+kqX94g92kjkXZV5va5KjkJL6WOibvYRaii8p/tCrcVSphqZsKVgc7KWCrUc1Db3Qp085sbPx9VSOzouBYjLFsLnKxJallz06yK2nhS1Sq2IQ06de16gqGuadQKF46vdBKEBbkC6lQcxey0jQ3drVMjTrVKRTuXQ5GxIAdCLOAnZjOXjZO3quHN24ad9XRT+zOf7akM6DffTLqJBbvbPFOqtBivFwhxZgwdGJtUVh7ymxz8DOj/wAjKad8tM+lvGWThNS+xtupXPZkcFUAMUJBup0em4yqIeo9bSWnCm2zTo11OGqcSUqgZbaIS1qgQ8qbAnLS97azuOHrq6h1IKkXBGYIMspWSImjV2vSVil2JX3uBHdVPRmVSFPgTKjeifFKqrAMpDKcwQQQR1BE+4CIiAiIgIiICIiAiIgIiICIiAiIgIiICIkbtDabK4o0qfaVLBmBbgRFJIBd7HUg2ABJsdIElEru0d46lDhNXDEhtOxqCocte6yrI7avtDoU6LVFo1mqAC1JqbKTf63FYi3kYHM958G+PbaPESK9DFFVQ/WogFVA8MpEblV0w/GQ9dSbXpLSDtcXF1Zm4V5i7X8jMO16rVMSB2wLv33em7AsajFn4jkQRkOA6WEtu72wkOSA56k5k+ZmJy103lHbizrw02zNO/G9Q2tetVOZ1NgLW5WkxgthcdgFCqMgALADwk3sjdwKASJZcPhAo0mpMZ1XcPuwoGk9r7sqRpLVEo5LtDdZsOzPRRGVrcdF7hWIJKulRe9RqC5sy5Z5gyL2rvFXNCphPpkWopVu2S9RVOoSvSJD3FxcqDYzsuJwisMxK5tHdsNe0z+V1wTZuCp8T1boqIrKbPxNdhnxHynSvZbvUlKglCrVACi6szDhKE5WY8xpYzT257OqbcR4CCQc1JyJ5gaSI3D3L7aq2HrcKUqLsSR/OVmAAAN9EGZtzJ8BJJYvbtVTGdsPo2tT0NQHNraqh5DkW+HUamyds4d2FCkSDwcaAoyB6d7cdO47y3IzHUdZsYDZiUV7NOLgtbhZiw8xfS/hlKhtPZtLZFCttINWrmjT7PD06rgrSWo6KKaEC/DxcGpNgMpplduwKktTIUnNl+ox6kcj94Z9b6TMm0V0qfRn73u+j6H5HwnL9nb07Ww9XD1MetBsNi+HKmQHoh2VQT1Cl04tddcp1EyjZWoCLggjqDeYquMpr71RF82UfqZptg6ZzNND5op/hMlKkq+6oX8IA/SBk/lOj/WKfLP9J5/KtHnVQebBf1jjn1eBnpVVYXVgw6ggj4ifcjamBpseIoOL7S91vR1sR8Y4KqZo/GPsVf4VALj8waBJRNXC45XPAQUcZlGyNuqkZMPEEzagIiICIiAiIgIiICIiB4xtmdBKftfeBMHh2xtRGbtXDWUZhWKpSFiRbu8NxcZ8XrZNrG6Cl/WsE/LYl/3A3qRMW0cAtVSjAEEFSCAylTqrKdR8I+LENiez2lgnp06vCKyWWomds88j4jhZT4iZtj7r06WH7Cpw1LszsQvZqCx4itNQTwJ90Hr1n3gMAmGpilTpimi5qVzUXOueY9cvGSlHEhrDQ29D4g85mXyWeFe2lu5s5Az1aSKGIFyWWxtolje5tew6TDgthCkBVwjrUpsLqrtqPuVR/Eesy73CnUalTXH0sLiUYvT4zTa4ZShDUmIuCDkbg3ElN3NjDCYdMOHL8PESxFrs7l2NuQuxsOk0j62VthXupBVlPCyNbiVhyNvTTIggyZU3nIMXtVhtavwnJnA/7ainf4qZ1bZz3QGBtREQE8InsQI3aov9EmTsM214F0L268gOvgDKj/0riaeMpvQrBcMppsQXqcY4eLtUKW4KgqXF2bMWHSWWnibYqrTcW4wjUm5MApDUwftAhmtzDkjQ2rm++2MZTdKOEwxqs9+9YlEtY97lnnmxAy6yW4si2viANM/Hl6mRvZrilelURnoVEsxccIYki3Z6G1rm+nu25zXwPHXphqoBRVHGEzFaoB3lXrSBuPvH7o70Zurvk2KrpSvTdalFqp7MMGw7KVBo17k3bva2XMaTMu9rWGt7OKQqdvTcs1rcFW7IwJuQeS3azHhAFxpYmWfd7BVKOHSjVq9q63HGfs8R4VvzsthfwklBm8ZeTT2ptahh1469VaY+8c/Qamc+329p3ZscLgFFSrmGrHOmh07v2j/wX1lAw+wa+Mqdriaj1nJv3vdHkukDpGP9sWAU8NJa1Y9UQBfiTf5TUT2x0754GvbqCCfhaa2y9xjYd23kLSZXcQW0gbmyfajs+qwR3egx0FdCgPk4uPjaXSjVVgGUhlOYIIII6gjWcyx24mRyuPEXkbsvC4vZz8WHY9nfvUHJNJvIa028V9QYHXcRh1cWYaZggkFT1VhmD4iYqGMamwp1jfiNqdSwAY/ZcDJX8sm5W0mnu5vBSxdMsl1dbCpTb36bW0PUHkwyMksRQV1KMLqRYj/mh8YG5Ei9mYhlY4eobsoujn+kp6Bj94aN42OXEBJSAiIgIifLuALkgAakmw+MD6iarYy/uqW8T3V+JzPoDMbVKh+sF/CLn4tl8oG9Ejuz6s5/OR8lsJ4cOh1UHzz/AFgZG71YnlTXh/M9mPqAqf4psGaGAUJUqoABfgqWGQzXs8v+385vMfWUUVNgY39vWqRTCJXeqcSHPa1KLcVsM9O2i3CjO1lB1lwTAqHFS7ZX4VuOEE5EgAXva4zJtfKbQaJMXX552jSCfylTxGBOKxlbE1FpVfeKZrwAZ8QFmBHD1ANhadz3bwz0sJQpVCS6U1VuI3bIaMeZAsL+EjdlbpCjjK2LNZnFSoaq02UWp1GXhLBueVwPOWWSFcO2E3a41qn2iGPmw4z82M7ds9bIJxrdjCmljqlJtUcr/hPD/CdowfuiVGeIiAiIgQe8ezDVW6khhmCMjcZjMeIB8xIvZmPSuRhsXTQ1l93jUFagHNbiwbqvO1xzAt5Eg9vbBSst7WYZgjIgjMEEaGBIhQBYfATFQwlNCzIiqWN2KqAWPViBmfOQWzNtPTYYfFmx0SsclbotTkreOh8DrY4Hxrl0nLvaXvkzFsBhWy92s6nU6GmpHLr100vex+0jej9lo9jTP09UWFtUQ5F/AnQep5ShblbuGq4dhlA+90d0GqWZhlOr7I3eSmB3RN7ZWzlpqABJGBjSiBoJ92nsQPlkB5TTxezEcZgTeiBQ9obDfD1RisPk65EaComppv4HkeRsZbcDilq01qrowvY6g6FT4g3B8RNutSDCxkXs+j2bvT5N3x56N/pPqYHm20PB2yC70TxgDVlH84njxLe33gp5STwlcOoYG4IBB6g6GfEiN1H4Uah/U1HpAdEVj2f/AOZSBYIiICaOOH0lMnQhh+bJgR6B5vTRxb8Tqg+p3yfEgqo+BY+g6wBiJy/fLfRqxOFwp7mjONX8ui/rKLFvFv8A0MOTTpjtnGtjwoD04rEsfIEeMrP/AF9j6p+jpoo5WQn5sx/SY92dyWqEPUub9Z0jZ27lOmB3RIOeNtjajMtS9mW4BVKYup1U93MZA+kkqW+2Lp2FVFP2r02v6cBAnRVwCDkJr4jY1NtVECrYP2hUjlUpEeKEMf8ABqJZNlbbw+I/maquRqt7OPNTmJoYndGi31BIDaW4K3D0yVZfdYEhh5EZiBfotKPsvb2JwhFLG3qUtBiAO+n96o95fvDMc7y8U3DAMCCCAQQbgg5gg8xAoG+Wy+wxSY9R3Klkq+DjJWPgRYeYHWXbY2KDoCOk92jglrU3ouLq6kH+BHiDYjylP3Sxz0aj4Sqe/TbhPiOTDwIsfWBf4nyjXF59QEREBERAidsbPp1SqMoYE3YEZcK6g+ZIHkTIPF7pUkV3p4rE4ZQCx7OseFQMyQHBsLXyljpm7M/U8I/Clx/m4j6iVf2m4408EUU51nWn+XN2+IW35oHL9m4Cpiq5ZnqVBxHv1TdyL93jPUCw9J2ndzZC0kGUrfs/2OFQMRnOgqtsoHsREBERAREQE08Sv0iH8Q9LX/gJuTWrnvqOgJ+JAH+r4QPTK5sepbHYxPv0n/xYemP9MsUpewcTx7Tx/RalJB+Sgl/mTAvsTwRAMbZmR9AZcR1Y8R9dB6Cw9Js7QP0ZH2rL6MwU/rMZgUj2l7cNOmMJTPfrC79RTva35iCPJT1kbuPuve1VxcnPOaFdDjNoVahzUOVXpwp3RbzAv6mdV2ThAiAAcoGxhsMEFgJniICIiAnhE9iBrYnBI4sQJEbOw5wrij/QOfo/7NznwD7jZ26HL62VgmHF4cVEKHmMjzB5EeINj6QBnPvaAnYYmhi1yFT6N/Nc1PmRcfkl9wlXjRWOpHeHRhkw+IMqftTocWz6j86TU6g8LOFP7rNAsWwsZ2lMHwkpKJ7O8dxUlF5e4CIiAmLFVOFGYagG3nbL5zLNbH+6q9XX908Z+SmBjpU+FQvQAfAWlL9qNK9PDHkK1j6oSP0Mu0gd98H2mEcgZ0itUfkPe/cLwNzdakBSXyk7ILdWpekvlJ2AiIgIiICIiAJmjQbivU+1p+Ae78cz+ae4p+M9kNB/OH5hPXU+HnMsD4r1gis7GyqCzHoALk/Cc79ld6oq4ptcRWqVvR2JA+Fpv+1na5pYQYWmfpcW3YrbUJ/SN5Wy/NJbcfZopUUUDIKAPQQLUIiIGrjj7g6uPkrN+oExVGsCegJ+An3tLJVbpUT948H+qeWgUD2bYLiRah1IvedLUWEp3s8ocFE0+dKpUpN5o5H+0uUBERAREQEREBERA0dnjusP7Sr86jH+MgvaR/7Zi/7pv1FpO7OH0YP2i7+juWHyIlV9reK4NmVhzqFKY/M63+QMCB9l1XugTqq6Tk/svTKdYXSB7ERASP2vXVOzZ2VV4/eYhQCabgXJyGskJixNBXUoyhgRmCAQfQwMCMCLg3HUZj4z0iU3E7olH7SgxpEG44O7mPDQjwli2PtA1FKuAtZMqijTwdfun5G45QNXY1L9nqHD/U1peKfZ81vb4HnLKJG4vCioLHIg3Vhqp6j/AG5xg8aQRSq91vqn6r+Knr4aj5wJKIiAiIgJrbQxHAuRAZiFW/U87c7C5t4T3FYtUyzLH3UXNj4+A8TlI6th2e7uRx8rZrTsQQFvrmASeduQsAFapbTxGIesuGrLRShUNIFk7RqlQAMzuSchduWeslN094TiMO9SuFR6DvTrEHuXp5l1J0W2fxlI3i2ctDEVMR+11cF2pvVU0nq0nbm1J1uM+hAIJOki8dig9EbMwfH2Td6vVYFXqs1sraqoAHicuhvFbOBxLbU2k2Msexp/R4cH7IOb26k3PrOxbPocKASrbk7vrQpqAtrAS5ASo9iIgYMbSLU2Ua27v4hmp+IE16VQMoYaMAR5HOb8jkXhdqfW7p4gnvj0Y/BxAi8EvYY1x9TFAOvQV6a2dfzIA35WllkVtLAiqnBcqQQyONUdTdXHl05i45zY2bjC44Kg4aq++vI9HTqh5H0OYMDdiIgIiICIiAmvjXPDwjV+6PC+p9Bc/CZqjhQWJsBmSZqU7se0ItcWUHUL4+JyJ8gOUDMuQsNBpOTe27afE+FwS63NZ/Id1PmWnVMTiFRGqOQFUFmJ0AAuTPz2MY20doVcWQeF24aYPKkmS/GB032cYLhpg2nQxILdfBcFMZcpOwEREBERA8IkLtjZhJFameGol+FrcjqrDmptmP4gSbnhECC2bthah7Jx2dYaoefjTb6w+Y5iSFairqVZQwOoIuJpba2ElYZjPUHmD1B5GQf7ZjMLkR+0Ux9o8NQDwqWPF6i/jAsdKhUT+bqkr9moOMD8L3DD1LTMMVU501P4an+6iQWG3wwxyqFqLdKqkD/GLr85LYfadGpmlam34ain9DAzNjKnKkPzVLf5VMxsaze84QdKYz8i7cvJQfGZDWXXiX4iaGK3gwtP38RSHgHVj6KtyZBvUqAW9hrqcyT4sxzPrPvTM+sqeN38o6UKdSq3I24E+Jz+UiK7YzGm1Q8FM/0aXAP4jqZVSG8m9JYnDYM8TnJqozVPBOrePKfe6m7ATvNmTmScySdST1knsHdlKQGUs9KmFFhCPKNIKLCZIiAiIgJq7Qwxde4bOp4kJ0v0b7pFwfA5ZgTaiBHYXEioCQLFTwup1RgASp+IN9CCCMjPqtQDW1BHusMmXyP8DkeYnxtLBvftqJAqAWIPu1FH1HtpzswzF+YJB+MDtBat1zWovv02ydfEjmvRhcHrAypiaqZOoqD7SWDfmpsfmpN+gn3T2tSOYbLS5VgARqCSLCR+3cTwU72JB4rgalUpvVZR5hCvrKBjtsYpcJ+3ri2FRCt6ICiit2A7MU7aAEZk5yW4sjrNKsrC6sGHUEH9J9yD2HWTFYejiWpKGq00exAuOJQbX1m7+yLyLjyqVB+jSo35rPjlvZbu3RM7eZ0X1ImP9iQ6gt+Nmf8AzEzOqgCwyA6QMIpMxDVLZZqgzUHqT9Y/IdOcymfNasFUsxAA1JNgPMzlG/HtLLlsHs/vMbrUr/VTrw9T4f8ACGr7Wt7jWb+SsM1xcftDqcgL/wA2CP8Anwm57Pt3OEKSOkhdy90iW4muSTdmbMsTqSZ2bZGzxTUACBu4enwgCZYiAiIgIiICIiAmKrh1bUTLECExm79N/qiQeK3IpN9UfCXeIHOv+gaf2R8JtYfchB9US92iBXcHuzTT6okzQwSroJsxA8AnsRAREQEREBERASP2nspKtibhl911JV1P3WGY8uckIgVLaIxVNQCor8JDK2SVLjKzL7jggsptw5Mecpf8lbMaqHxDYnDj/wCPVBNIZ3Kq4Ujgv4idfZQdZo4vZSOLED4QNDZ+8OB4QtPFYfhAAUCrTFlAsBa+U2qm3sKoucTRA6mrT/3ld2h7PsI970KefRQP0kRU9l2F5UF+cCxbQ9oWzaIu2MpnwQmof3AZU9p+2KmbrgsLVrtyZxwJ52GZ+Im9hvZxh092ig/Lf9ZMYXc9F+qPQQOXY1tp7RP/AKmsUpn+hpd1bdCR/wCfGWjdnccKB3bAcrToeE2EichJSnRC6CBo7M2WtMAASSiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgItEQPLT2IgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiIH//Z'
# response = requests.get(url, stream = True)
# img = Image.open(response.raw)
image_url ="data:image/jpeg;base64,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"
if re.match(r"^data:image\/[a-zA-Z]+;base64,", image_url):
    
    base64_str = image_url.split(",", 1)[1]  
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    print("trường hợp data")
    print(image)
    image.show()
else:
    
    response = requests.get(image_url)
    response.raise_for_status()  
    image = Image.open(BytesIO(response.content)).convert("RGB")
    print("truong hop http:")
    print(image)
    image.show() 
    # .overlay {
    #     position: fixed;
    #     top: 0;
    #     left: 0;
    #     width: 100%;
    #     height: 100%;
    #     background: rgba(0, 0, 0, 0.5);
    #     z-index: 500;
    #     display: none;
    # }
package org.leadersofdigital.hack1;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import opennlp.tools.tokenize.SimpleTokenizer;
import opennlp.tools.tokenize.Tokenizer;
import org.apache.commons.lang.StringUtils;

/**
 *
 * @author yurij
 */
public class AddressTransformer {
    
    private static final Pattern ADDR_REGEX = Pattern.compile(".*?(((кв|к|стр|д|дом)\\.?\\s[\\d]{1,4}\\/?[\\dа-яА-Я]{0,5})|([а-яА-Я]{4,}\\s[\\d\\-\\/]{1,5}))");
    private static final Pattern MISSING_DATA_REGEX = Pattern.compile("(?<missing>(корп|стр)\\s)[а-яА-Я]"); 
    
    private final Tokenizer tokenizer = SimpleTokenizer.INSTANCE;
    
    private final List<String> metros;
    private final Set<String> badWords;
    private final Set<Character> punctuation; 
    
    public AddressTransformer() throws IOException {
        metros = Files.readAllLines(Paths.get("../metro.csv"));
        badWords = new HashSet<>(Files.readAllLines(Paths.get("../bad_words.csv")));
        punctuation = "«!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~»".chars().mapToObj(e->(char)e).collect(Collectors.toSet());
    }
    
    private String clearAddress(String addr) {
        Matcher matcher = ADDR_REGEX.matcher(addr);
        return matcher.find() ? matcher.group(0) : addr;
    }

    private boolean isOnlyPunctuation(String s) {
        for (int i = 0; i < s.length(); i++) {
            if (!punctuation.contains(s.charAt(i))) {
                return false;
            }
        }
        return true;
    }
    
    private String transformWord(String s) {
        Integer num = romanToDecimal(s.replace('м', 'm').replace('х', 'x'));
        if(num != null)
            return num.toString();
       
        if (isOnlyPunctuation(s))
            return null;

        if (s.length() == 3 && s.endsWith("ао")) // цао
            return null;

        if (badWords.contains(s))
            return null;

        return StringUtils.strip(s, ".'\"`-");
    }

    private static Integer romanToDecimal(String romanNumber) {
        int decimal = 0;
        int lastNumber = 0;
        String romanNumeral = romanNumber.toUpperCase();
        for (int x = romanNumeral.length() - 1; x >= 0 ; x--) {
            char convertToDecimal = romanNumeral.charAt(x);

            switch (convertToDecimal) {
                case 'M':
                    decimal = processDecimal(1000, lastNumber, decimal);
                    lastNumber = 1000;
                    break;

                case 'D':
                    decimal = processDecimal(500, lastNumber, decimal);
                    lastNumber = 500;
                    break;

                case 'C':
                    decimal = processDecimal(100, lastNumber, decimal);
                    lastNumber = 100;
                    break;

                case 'L':
                    decimal = processDecimal(50, lastNumber, decimal);
                    lastNumber = 50;
                    break;

                case 'X':
                    decimal = processDecimal(10, lastNumber, decimal);
                    lastNumber = 10;
                    break;

                case 'V':
                    decimal = processDecimal(5, lastNumber, decimal);
                    lastNumber = 5;
                    break;

                case 'I':
                    decimal = processDecimal(1, lastNumber, decimal);
                    lastNumber = 1;
                    break;
                
                default:
                    return null;
            }
        }
        return decimal;
    }

    private static int processDecimal(int decimal, int lastNumber, int lastDecimal) {
        if (lastNumber > decimal) {
            return lastDecimal - decimal;
        } else {
            return lastDecimal + decimal;
        }
    }
    
    
    public String tranfsorm(String address) {
        address = clearAddress(address.toLowerCase());
        for (String s : metros) {
            address = StringUtils.replace(StringUtils.replace(address, "м." + s, ""), "м. " + s, "");
        }
        List<String> words = new ArrayList<>();
        for (String w : tokenizer.tokenize(address)) {
            if (w.contains(".")) {
                for(String p : w.split("\\.")) {
                    if(p.length() > 0) {
                        words.add(w);
                    }
                }
            }
            else {
                words.add(w);
            }
        }
        
        List<String> words2 = new ArrayList<>();        
        for(String w : words) {
            String r = transformWord(w);
            if(r != null)
                words2.add(r);
        }
        
        String res = String.join(" ", words2);
        
        Matcher matcher = MISSING_DATA_REGEX.matcher(res);
        if(matcher.find()) {
            res = StringUtils.replace(res, matcher.group("missing"), "");
        }
        
        return res;
    }
}
